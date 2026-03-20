// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for unique element detection
//!
//! This module contains the multi-pass GPU execution:
//! 1. Pass 1: Mark unique values using hash table (parallel)
//! 2. Pass 2: Compute prefix sum of unique flags (parallel)
//! 3. Pass 3: Read unique count from prefix sum
//! 4. Pass 4: Compact unique values (parallel)

use super::Unique;
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
fn compute_prefix_sum_gpu(
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
    ComputeDispatch::new(device, "unique_prefix_local_scan")
        .shader(Unique::prefix_sum_shader(), "local_scan")
        .uniform(0, &params_buffer)
        .storage_read(1, mask_buffer)
        .storage_rw(2, &prefix_sum_buffer)
        .storage_rw(3, &scratch_buffer)
        .dispatch(n_groups.max(1), 1, 1)
        .submit()?;

    // Pass 2: add workgroup offsets (only when n_groups > 1)
    if n_groups > 1 {
        ComputeDispatch::new(device, "unique_prefix_add_offsets")
            .shader(Unique::prefix_sum_shader(), "add_wg_offsets")
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

/// Read only the last element of a u32 buffer
fn read_buffer_u32_last(
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
    device.submit_commands(Some(encoder.finish()));

    let result_data: Vec<u32> = device.map_staging_buffer(&staging_buffer, 1)?;
    Ok(result_data[0])
}

/// Execute unique operation
pub(super) fn execute(unique: Unique) -> Result<Tensor> {
    let device = unique.input().device();
    let input_size: usize = unique.input().len();

    // Use hash table size - large minimum to avoid collisions (hash stores only occupancy, not value)
    let num_buckets = (input_size * 32).next_power_of_two().clamp(8192, 65_536);

    // Create hash table (atomic u32)
    let hash_table_buffer = device.create_buffer_u32(num_buckets)?;

    // Create unique flags buffer
    let unique_flags_buffer = device.create_buffer_u32(input_size)?;

    // Initialize hash table to zeros
    let zeros = vec![0u32; num_buckets];
    device
        .queue
        .write_buffer(&hash_table_buffer, 0, bytemuck::cast_slice(&zeros));

    // Initialize flags to zeros
    let zeros_flags = vec![0u32; input_size];
    device
        .queue
        .write_buffer(&unique_flags_buffer, 0, bytemuck::cast_slice(&zeros_flags));

    let params = super::UniqueParams {
        input_size: input_size as u32,
        num_buckets: num_buckets as u32,
        _pad1: 0,
        _pad2: 0,
    };

    let params_buffer = device.create_uniform_buffer("Unique Params", &params);

    // Step 1: Mark unique values
    let caps = DeviceCapabilities::from_device(device);
    let workgroups = caps.dispatch_1d(input_size as u32);

    ComputeDispatch::new(device, "unique_mark")
        .shader(Unique::wgsl_shader(), "mark_unique")
        .uniform(0, &params_buffer)
        .storage_read(1, unique.input().buffer())
        .storage_rw(2, &hash_table_buffer)
        .storage_rw(3, &unique_flags_buffer)
        .dispatch(workgroups, 1, 1)
        .submit()?;

    // Step 2: Compute prefix sum of unique flags to determine output positions
    let (prefix_sum_buffer, unique_count) =
        compute_prefix_sum_gpu(device, &unique_flags_buffer, input_size)?;
    let unique_count = unique_count as usize;

    if unique_count == 0 {
        return Ok(Tensor::new(vec![], vec![0], device.clone()));
    }

    // Step 4: Compact unique values using GPU shader
    let output_buffer = device.create_buffer_f32(unique_count)?;

    let workgroups = caps.dispatch_1d(input_size as u32);

    ComputeDispatch::new(device, "unique_compact")
        .shader(Unique::wgsl_shader(), "compact_unique")
        .uniform(0, &params_buffer)
        .storage_read(1, unique.input().buffer())
        .storage_rw(2, &hash_table_buffer)
        .storage_rw(3, &unique_flags_buffer)
        .storage_read(4, &prefix_sum_buffer)
        .storage_rw(5, &output_buffer)
        .dispatch(workgroups, 1, 1)
        .submit()?;

    let output_data = crate::utils::read_buffer(device, &output_buffer, unique_count)?;
    Ok(Tensor::new(output_data, vec![unique_count], device.clone()))
}
