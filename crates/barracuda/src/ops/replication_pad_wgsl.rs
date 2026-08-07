// SPDX-License-Identifier: AGPL-3.0-or-later
//! Replication Pad - Pad by replicating edge values - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its padding parameters
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Replication pad operation
pub struct ReplicationPad {
    input: Tensor,
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
}

impl ReplicationPad {
    /// Create a new replication pad operation
    #[must_use]
    pub fn new(input: Tensor, padding: (usize, usize, usize, usize)) -> Self {
        Self { input, padding }
    }

    /// Execute the replication pad operation
    /// # Errors
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

        let (pad_left, pad_right, pad_top, pad_bottom) = self.padding;
        let out_height = in_height + pad_top + pad_bottom;
        let out_width = in_width + pad_left + pad_right;

        let output_size = batch * channels * out_height * out_width;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReplicationPad Output"),
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
            pad_top: u32,
            pad_left: u32,
        }

        let params = Params {
            batch: batch as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            pad_top: pad_top as u32,
            pad_left: pad_left as u32,
        };

        let params_buffer = device.create_uniform_buffer("ReplicationPad Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
        let workgroups_x = (out_width as u32).div_ceil(optimal_wg_size);
        let workgroups_y = (out_height as u32).div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "ReplicationPad")
            .shader(include_str!("../shaders/tensor/replication_pad_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, 1)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![batch, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply replication padding (NCHW format)
    /// # Arguments
    /// * `padding` - (left, right, top, bottom) padding amounts
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn replication_pad_wgsl(self, padding: (usize, usize, usize, usize)) -> Result<Self> {
        ReplicationPad::new(self, padding).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_replication_pad_simple() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1x1x2x2 input
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 1, 2, 2], device);

        // Pad by 1 on all sides
        let output = input.replication_pad_wgsl((1, 1, 1, 1)).unwrap();

        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        assert_eq!(output.to_vec().unwrap().len(), 16);

        // Check edge replication
        let result = output.to_vec().unwrap();
        // Top-left corner should replicate input[0,0] = 1.0
        assert_eq!(result[0], 1.0);
    }

    #[tokio::test]
    async fn test_replication_pad_asymmetric() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1x1x2x2 input
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![1, 1, 2, 2], device);

        // Asymmetric padding
        let output = input.replication_pad_wgsl((2, 1, 0, 1)).unwrap();

        assert_eq!(output.shape(), &[1, 1, 3, 5]);
        assert_eq!(output.to_vec().unwrap().len(), 15);
    }
}
