// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clip Gradient by Norm - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Two-pass operation:
//! 1. Compute total norm (parallel reduction)
//! 2. Clip gradients based on computed norm
//!
//! Shader: f64 canonical (downcast to f32 at compile)

use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::Result;
use crate::tensor::Tensor;

/// Clip gradient by norm operation
pub struct ClipGradNorm {
    gradients: Tensor,
    max_norm: f32,
}

impl ClipGradNorm {
    /// Create a new clip gradient by norm operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(gradients: Tensor, max_norm: f32) -> Result<Self> {
        if max_norm < 0.0 {
            return Err(crate::error::BarracudaError::invalid_input(
                "max_norm must be non-negative",
            ));
        }
        Ok(Self {
            gradients,
            max_norm,
        })
    }

    /// Execute the clip gradient by norm operation (2-pass)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.gradients.device();
        let size: usize = self.gradients.shape().iter().product();

        let input_buffer = self.gradients.buffer();

        let num_workgroups =
            size.div_ceil(crate::device::capabilities::WORKGROUP_SIZE_1D as usize) as u32;
        let norm_buffer_size = num_workgroups.max(1) as usize;
        let norm_buffer = device.create_buffer_f32(norm_buffer_size)?;

        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            max_norm: f32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            max_norm: self.max_norm,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ClipGradNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let shader_source = include_str!("../shaders/gradient/clip_grad_norm_f64.wgsl");
        let mut batch = BatchedComputeDispatch::new(device);

        batch.push(
            ComputeDispatch::new(device, "ClipGradNorm Norm")
                .shader(shader_source, "compute_norm")
                .uniform(0, &params_buffer)
                .storage_read(1, input_buffer)
                .storage_rw(2, &norm_buffer)
                .storage_rw(3, &output_buffer)
                .dispatch(num_workgroups, 1, 1),
        )?;

        if num_workgroups > 1 {
            batch.push(
                ComputeDispatch::new(device, "ClipGradNorm Norm Final")
                    .shader(shader_source, "compute_norm_final")
                    .uniform(0, &params_buffer)
                    .storage_read(1, input_buffer)
                    .storage_rw(2, &norm_buffer)
                    .storage_rw(3, &output_buffer)
                    .dispatch(1, 1, 1),
            )?;
        }

        batch.push(
            ComputeDispatch::new(device, "ClipGradNorm Clip")
                .shader(shader_source, "clip_gradients")
                .uniform(0, &params_buffer)
                .storage_read(1, input_buffer)
                .storage_rw(2, &norm_buffer)
                .storage_rw(3, &output_buffer)
                .dispatch_1d(size as u32),
        )?;

        batch.submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.gradients.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clip_grad_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[3.0, 4.0], vec![2], device).unwrap();

        let clipped = ClipGradNorm::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        // Original norm = 5, should be clipped to norm = 1
        let norm: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1); // Allow some tolerance for atomic operations
    }

    #[tokio::test]
    async fn test_clip_grad_norm_no_clip() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[0.1, 0.2, 0.3], vec![3], device).unwrap();

        let clipped = ClipGradNorm::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        // Norm ≈ 0.374, should not be clipped
        let norm: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm < 1.0);
    }

    #[tokio::test]
    async fn test_clip_grad_norm_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[0.0, 0.0, 0.0], vec![3], device).unwrap();

        let clipped = ClipGradNorm::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_clip_grad_norm_large() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..1000).map(|i| (i % 10) as f32).collect();
        let gradients = Tensor::from_data(&data, vec![1000], device).unwrap();

        let clipped = ClipGradNorm::new(gradients, 100.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result.len(), 1000);
        let norm: f32 = result.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!(norm <= 100.0 + 1.0); // Allow tolerance
    }

    #[tokio::test]
    async fn test_clip_grad_norm_invalid() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(ClipGradNorm::new(gradients, -1.0).is_err());
    }
}
