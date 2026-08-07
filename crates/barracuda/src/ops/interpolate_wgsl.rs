// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interpolate - Resize using bilinear interpolation - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows target size
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Interpolate operation - Resize using bilinear interpolation
pub struct Interpolate {
    input: Tensor,
    out_height: usize,
    out_width: usize,
}

impl Interpolate {
    /// Create a new interpolate operation
    #[must_use]
    pub fn new(input: Tensor, out_height: usize, out_width: usize) -> Self {
        Self {
            input,
            out_height,
            out_width,
        }
    }

    /// Execute the interpolate operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Assume NCHW format
        let batch = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        let output_size = batch * channels * self.out_height * self.out_width;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Interpolate Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch: u32,
            channels: u32,
            in_height: u32,
            in_width: u32,
            out_height: u32,
            out_width: u32,
        }

        let params = Params {
            batch: batch as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: self.out_height as u32,
            out_width: self.out_width as u32,
        };

        let params_buffer = device.create_uniform_buffer("Interpolate Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
        let workgroups_x = (self.out_width as u32).div_ceil(optimal_wg_size);
        let workgroups_y = (self.out_height as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "Interpolate")
            .shader(include_str!("../shaders/misc/interpolate_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![batch, channels, self.out_height, self.out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Resize tensor using bilinear interpolation (NCHW format)
    ///
    /// # Arguments
    ///
    /// * `out_height` - Target height
    /// * `out_width` - Target width
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn interpolate_wgsl(self, out_height: usize, out_width: usize) -> Result<Self> {
        Interpolate::new(self, out_height, out_width).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_interpolate_upscale() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1x1x2x2 input
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 1, 2, 2], device);

        let output = input.interpolate_wgsl(4, 4).unwrap();

        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        assert_eq!(output.to_vec().unwrap().len(), 16);
    }

    #[tokio::test]
    async fn test_interpolate_downscale() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1x1x4x4 input
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::new(data, vec![1, 1, 4, 4], device);

        let output = input.interpolate_wgsl(2, 2).unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.to_vec().unwrap().len(), 4);
    }
}
