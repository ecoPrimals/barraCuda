// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for `NonZero`
//!
//! This module contains the GPU execution logic for finding non-zero elements,
//! including mask conversion, prefix sum computation, and index compaction.

use super::{NonZero, NonZeroParams};
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

impl NonZero {
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
        ComputeDispatch::new(device, "nonzero_prefix_local_scan")
            .shader(Self::prefix_sum_shader(), "local_scan")
            .uniform(0, &params_buffer)
            .storage_read(1, mask_buffer)
            .storage_rw(2, &prefix_sum_buffer)
            .storage_rw(3, &scratch_buffer)
            .dispatch(n_groups.max(1), 1, 1)
            .submit()?;

        // Pass 2: add workgroup offsets (only when n_groups > 1)
        if n_groups > 1 {
            ComputeDispatch::new(device, "nonzero_prefix_add_offsets")
                .shader(Self::prefix_sum_shader(), "add_wg_offsets")
                .uniform(0, &params_buffer)
                .storage_read(1, mask_buffer)
                .storage_rw(2, &prefix_sum_buffer)
                .storage_rw(3, &scratch_buffer)
                .dispatch(1, 1, 1)
                .submit()?;
        }

        // Total = scan_out[N-1] + flags_in[N-1] (exclusive scan + last flag)
        let scan_last = Self::read_buffer_u32_last(device, &prefix_sum_buffer, size)?;
        let flags_last = Self::read_buffer_u32_last(device, mask_buffer, size)?;
        let total = scan_last + flags_last;

        Ok((prefix_sum_buffer, total))
    }

    /// Convert f32 mask to u32 mask on GPU
    pub(super) fn convert_mask_gpu(
        device: &Arc<crate::device::WgpuDevice>,
        input_buffer: &wgpu::Buffer,
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

        ComputeDispatch::new(device, "nonzero_mask_convert")
            .shader(Self::mask_convert_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_rw(2, mask_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        Ok(())
    }

    /// Convert u32 indices to f32 on GPU
    pub(super) fn convert_u32_to_f32_gpu(
        device: &Arc<crate::device::WgpuDevice>,
        u32_buffer: &wgpu::Buffer,
        f32_buffer: &wgpu::Buffer,
        size: usize,
    ) -> Result<()> {
        #[repr(C)]
        #[derive(Copy, Clone, Pod, Zeroable)]
        struct ConvertParams {
            size: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = ConvertParams {
            size: size as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer = device.create_uniform_buffer("U32 to F32 Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let workgroups = caps.dispatch_1d(size as u32);

        ComputeDispatch::new(device, "nonzero_u32_to_f32")
            .shader(Self::u32_to_f32_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, u32_buffer)
            .storage_rw(2, f32_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        Ok(())
    }

    /// Read only the last element of a u32 buffer (for getting prefix sum total)
    pub(super) fn read_buffer_u32_last(
        device: &Arc<crate::device::WgpuDevice>,
        buffer: &wgpu::Buffer,
        size: usize,
    ) -> Result<u32> {
        if size == 0 {
            return Ok(0);
        }
        // Read only the last element
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

    /// Execute `NonZero` operation (GPU-accelerated)
    ///
    /// **Deep Debt**: Efficient GPU implementation using prefix sum for compaction
    ///
    /// Returns: Tensor containing flat indices of non-zero elements
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input().device();
        let input_size: usize = self.input().len();

        // Step 1: Create boolean mask on GPU (convert f32 input to u32 mask)
        let mask_buffer = device.create_buffer_u32(input_size)?;
        Self::convert_mask_gpu(device, self.input().buffer(), &mask_buffer, input_size)?;

        // Step 2: Compute prefix sum on GPU
        let (prefix_sum_buffer, output_size) =
            Self::compute_prefix_sum_gpu(device, &mask_buffer, input_size)?;
        let output_size = output_size as usize;

        if output_size == 0 {
            // No nonzero elements
            return Ok(Tensor::from_buffer(
                device.create_buffer_u32(0)?,
                vec![0],
                device.clone(),
            ));
        }

        // Step 4: Execute nonzero shader to compact indices
        let output_buffer = device.create_buffer_u32(output_size)?;

        let params = NonZeroParams {
            input_size: input_size as u32,
            _padding: [0; 3],
        };
        let params_buffer = device.create_uniform_buffer("NonZero Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let workgroups = caps.dispatch_1d(input_size as u32);

        ComputeDispatch::new(device, "nonzero_compact")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.input().buffer())
            .storage_read(2, &prefix_sum_buffer)
            .storage_rw(3, &output_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        // Step 5: Convert u32 indices to f32 on GPU (for Tensor compatibility)
        let indices_f32_buffer = device.create_buffer_f32(output_size)?;
        Self::convert_u32_to_f32_gpu(device, &output_buffer, &indices_f32_buffer, output_size)?;

        Ok(Tensor::from_buffer(
            indices_f32_buffer,
            vec![output_size],
            device.clone(),
        ))
    }
}
