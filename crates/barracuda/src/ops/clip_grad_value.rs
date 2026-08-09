// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clip Gradient by Value - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/clip_grad_value_f64.wgsl");

/// Clip gradient by value operation
pub struct ClipGradValue {
    gradients: Tensor,
    clip_value: f32,
}

impl ClipGradValue {
    /// Create a new clip gradient by value operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(gradients: Tensor, clip_value: f32) -> Result<Self> {
        if clip_value < 0.0 {
            return Err(crate::error::BarracudaError::invalid_input(
                "clip_value must be non-negative",
            ));
        }
        Ok(Self {
            gradients,
            clip_value,
        })
    }

    /// Execute the clip gradient by value operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.gradients.device();
        let size: usize = self.gradients.shape().iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.gradients.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            clip_value: f32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            clip_value: self.clip_value,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ClipGradValue Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let caps = DeviceCapabilities::from_device(device);
        let workgroups = caps.dispatch_1d(size as u32);

        ComputeDispatch::new(device, "ClipGradValue")
            .shader(SHADER_F64, "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_rw(2, &output_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
        Ok(Tensor::new(
            output_data,
            self.gradients.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clip_grad_value_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[3.0, -4.0, 5.0, -6.0], vec![4], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result.len(), 4);
        assert!(result[0] <= 2.0 && result[0] >= -2.0);
        assert!(result[1] <= 2.0 && result[1] >= -2.0);
        assert!(result[2] <= 2.0 && result[2] >= -2.0);
        assert!(result[3] <= 2.0 && result[3] >= -2.0);
    }

    #[tokio::test]
    async fn test_clip_grad_value_no_clip() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[0.5, -0.3, 0.1], vec![3], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result[0], 0.5);
        assert_eq!(result[1], -0.3);
        assert_eq!(result[2], 0.1);
    }

    #[tokio::test]
    async fn test_clip_grad_value_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 0.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result, vec![0.0, 0.0, 0.0]);
    }

    #[tokio::test]
    async fn test_clip_grad_value_large() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..1000).map(|i| (i % 20) as f32 - 10.0).collect();
        let gradients = Tensor::from_data(&data, vec![1000], device).unwrap();

        let clipped = ClipGradValue::new(gradients, 5.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = clipped.to_vec().unwrap();

        assert_eq!(result.len(), 1000);
        assert!(result.iter().all(|&x| (-5.0..=5.0).contains(&x)));
    }

    #[tokio::test]
    async fn test_clip_grad_value_invalid() {
        let device = crate::device::test_pool::get_test_device().await;
        let gradients = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(ClipGradValue::new(gradients, -1.0).is_err());
    }
}
