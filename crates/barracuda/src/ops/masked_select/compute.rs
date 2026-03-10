// SPDX-License-Identifier: AGPL-3.0-only
//! GPU compute operations for Masked Select
//!
//! This module contains the GPU execution logic for masked select operation,
//! including prefix sum computation, mask conversion, and the main selection logic.

use super::MaskedSelect;
use crate::device::DeviceCapabilities;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ScanConfig {
    n: u32,
    n_groups: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Compute GPU prefix sum for boolean mask.
/// Returns (`prefix_sum_buffer`, `total_count`).
/// Uses exclusive scan: total = `scan_out`[N-1] + `flags_in`[N-1].
pub(super) fn compute_prefix_sum_gpu(
    device: &Arc<crate::device::WgpuDevice>,
    mask_buffer: &wgpu::Buffer,
    size: usize,
) -> Result<(wgpu::Buffer, u32)> {
    if size == 0 {
        let empty = device.create_buffer_u32(0)?;
        return Ok((empty, 0));
    }

    let prefix_sum_buffer = device.create_buffer_u32(size)?;
    let n_groups = (size as u32).div_ceil(WORKGROUP_SIZE_1D);
    let scratch_buffer = device.create_buffer_u32(n_groups as usize)?;

    let params = ScanConfig {
        n: size as u32,
        n_groups,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buffer = device.create_uniform_buffer("PrefixSum Params", &params);

    // Pass 1: intra-workgroup scan
    ComputeDispatch::new(device, "masked_select_prefix_local_scan")
        .shader(MaskedSelect::prefix_sum_shader(), "local_scan")
        .uniform(0, &params_buffer)
        .storage_read(1, mask_buffer)
        .storage_rw(2, &prefix_sum_buffer)
        .storage_rw(3, &scratch_buffer)
        .dispatch(n_groups.max(1), 1, 1)
        .submit()?;

    // Pass 2: add workgroup offsets (only when n_groups > 1)
    if n_groups > 1 {
        ComputeDispatch::new(device, "masked_select_prefix_add_offsets")
            .shader(MaskedSelect::prefix_sum_shader(), "add_wg_offsets")
            .uniform(0, &params_buffer)
            .storage_read(1, mask_buffer)
            .storage_rw(2, &prefix_sum_buffer)
            .storage_rw(3, &scratch_buffer)
            .dispatch(1, 1, 1)
            .submit()?;
    }

    // Total = scan_out[N-1] + flags_in[N-1] (exclusive scan + last flag)
    let scan_last = read_buffer_u32_last(device, &prefix_sum_buffer, size)?;
    let flags_last = read_buffer_u32_last(device, mask_buffer, size)?;
    let total = scan_last + flags_last;

    Ok((prefix_sum_buffer, total))
}

/// Convert f32 mask to u32 mask on GPU
pub(super) fn convert_mask_gpu(
    device: &Arc<crate::device::WgpuDevice>,
    input_mask_buffer: &wgpu::Buffer,
    mask_buffer: &wgpu::Buffer,
    size: usize,
) -> Result<()> {
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct MaskParams {
        size: u32,
        _pad1: u32,
        _pad2: u32,
        _pad3: u32,
    }

    let params = MaskParams {
        size: size as u32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let params_buffer = device.create_uniform_buffer("Mask Convert Params", &params);

    let caps = DeviceCapabilities::from_device(device);
    let workgroups = caps.dispatch_1d(size as u32);

    ComputeDispatch::new(device, "masked_select_mask_convert")
        .shader(MaskedSelect::mask_convert_shader(), "main")
        .uniform(0, &params_buffer)
        .storage_read(1, input_mask_buffer)
        .storage_rw(2, mask_buffer)
        .dispatch(workgroups, 1, 1)
        .submit()?;

    Ok(())
}

/// Read only the last element of a u32 buffer
pub(super) fn read_buffer_u32_last(
    device: &Arc<crate::device::WgpuDevice>,
    buffer: &wgpu::Buffer,
    size: usize,
) -> Result<u32> {
    if size == 0 {
        return Ok(0);
    }
    let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer U32 Last"),
        size: std::mem::size_of::<u32>() as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
        label: Some("Read Buffer Last Encoder"),
    });
    encoder.copy_buffer_to_buffer(
        buffer,
        ((size - 1) * std::mem::size_of::<u32>()) as u64,
        &staging_buffer,
        0,
        std::mem::size_of::<u32>() as u64,
    );
    device.submit_and_poll(Some(encoder.finish()));

    let result_data: Vec<u32> = device.map_staging_buffer(&staging_buffer, 1)?;
    Ok(result_data[0])
}

/// Execute the masked select operation (GPU compute)
pub(super) fn execute_masked_select(op: MaskedSelect) -> Result<Tensor> {
    let device = op.input().device();
    let input_size: usize = op.input().shape().iter().product();

    // Step 1: Create boolean mask buffer on GPU and convert f32 mask to u32
    let mask_buffer = device.create_buffer_u32(input_size)?;
    convert_mask_gpu(device, op.mask().buffer(), &mask_buffer, input_size)?;

    // Step 2: Compute prefix sum on GPU
    let (prefix_sum_buffer, output_size) =
        compute_prefix_sum_gpu(device, &mask_buffer, input_size)?;
    let output_size = output_size as usize;

    // Handle zero-size output
    if output_size == 0 {
        return Ok(Tensor::new(vec![], vec![0], device.clone()));
    }

    // Access input buffer directly (zero-copy)
    let input_buffer = op.input().buffer();

    // Create output buffer
    let output_buffer = device.create_buffer_f32(output_size)?;

    // Create uniform buffer for parameters
    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct Params {
        input_size: u32,
        _pad1: u32,
        _pad2: u32,
        _pad3: u32,
    }

    let params = Params {
        input_size: input_size as u32,
        _pad1: 0,
        _pad2: 0,
        _pad3: 0,
    };
    let params_buffer = device.create_uniform_buffer("MaskedSelect Params", &params);

    let caps = DeviceCapabilities::from_device(device);
    let workgroups = caps.dispatch_1d(input_size as u32);

    ComputeDispatch::new(device, "masked_select_main")
        .shader(MaskedSelect::wgsl_shader(), "main")
        .uniform(0, &params_buffer)
        .storage_read(1, input_buffer)
        .storage_read(2, &mask_buffer)
        .storage_read(3, &prefix_sum_buffer)
        .storage_rw(4, &output_buffer)
        .dispatch(workgroups, 1, 1)
        .submit()?;

    let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
    Ok(Tensor::new(output_data, vec![output_size], device.clone()))
}
