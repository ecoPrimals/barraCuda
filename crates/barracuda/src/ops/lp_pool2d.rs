// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lp Pool 2D - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Lp Pool 2D operation
pub struct LpPool2D {
    input: Tensor,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    p: f32,
}

impl LpPool2D {
    /// Create a new Lp Pool 2D operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        p: f32,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!("LpPool2D expects 4D tensor [B, C, H, W], got shape {shape:?}"),
            });
        }

        if p <= 0.0 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "p must be positive".to_string(),
            });
        }

        Ok(Self {
            input,
            kernel_size,
            stride,
            padding,
            p,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/pooling/lp_pool2d_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the Lp Pool 2D operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        // Compute output dimensions
        let out_height = ((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1;
        let out_width = ((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1;
        let output_size = batch_size * channels * out_height * out_width;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            channels: u32,
            in_height: u32,
            in_width: u32,
            out_height: u32,
            out_width: u32,
            kernel_size: u32,
            stride: u32,
            padding: u32,
            p: f32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            kernel_size: self.kernel_size as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            p: self.p,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LpPool2D Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let caps = DeviceCapabilities::from_device(device.as_ref());
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
        let workgroups_x = (out_width as u32).div_ceil(optimal_wg_size);
        let workgroups_y = (out_height as u32).div_ceil(optimal_wg_size);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device.as_ref(), "LpPool2D")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_rw(2, &output_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_lp_pool2d_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 8, 8], device.clone()).unwrap();

        let pooled = LpPool2D::new(input, 2, 2, 0, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(pooled.shape(), &vec![1, 1, 4, 4]);
    }

    #[tokio::test]
    async fn test_lp_pool2d_l1() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 4, 8], device.clone()).unwrap();

        let pooled = LpPool2D::new(input, 2, 1, 0, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(pooled.shape().len(), 4);
    }

    #[tokio::test]
    async fn test_lp_pool2d_invalid_shape() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device.clone()).unwrap();

        assert!(LpPool2D::new(input, 2, 2, 0, 2.0).is_err());
    }

    #[tokio::test]
    async fn test_lp_pool2d_invalid_p() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0; 16], vec![1, 1, 4, 4], device.clone()).unwrap();

        assert!(LpPool2D::new(input.clone(), 2, 2, 0, 0.0).is_err());
        assert!(LpPool2D::new(input, 2, 2, 0, -1.0).is_err());
    }

    #[tokio::test]
    async fn test_lp_pool2d_with_padding() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 1, 8, 8], device.clone()).unwrap();

        let pooled = LpPool2D::new(input, 3, 1, 1, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(pooled.shape().len(), 4);
    }
}
