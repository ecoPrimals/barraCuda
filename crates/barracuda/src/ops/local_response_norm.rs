// SPDX-License-Identifier: AGPL-3.0-or-later
//! `LocalResponseNorm` - Local Response Normalization (LRN)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Normalizes activations within local neighborhoods
//! Used in `AlexNet` and other early CNNs
//!
//! Formula: `y_i` = `x_i` / (k + alpha * `sum(x_j^2)` / size)^beta

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LocalResponseNormParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    size: u32,
    alpha: f32,
    beta: f32,
    k: f32,
}

/// Local response normalization (AlexNet-style LRN).
pub struct LocalResponseNorm {
    input: Tensor,
    size: usize,
    alpha: f32,
    beta: f32,
    k: f32,
}

impl LocalResponseNorm {
    /// Creates a new LRN. Size is the normalization window; alpha, beta, k are formula parameters.
    /// # Errors
    /// Returns [`Err`] if input is not 4D [B, C, H, W], or if size is zero.
    pub fn new(input: Tensor, size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "local_response_norm",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        if size == 0 {
            return Err(BarracudaError::invalid_op(
                "local_response_norm",
                "size must be positive",
            ));
        }

        Ok(Self {
            input,
            size,
            alpha,
            beta,
            k,
        })
    }

    /// Executes LRN and returns the normalized tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = LocalResponseNormParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            size: self.size as u32,
            alpha: self.alpha,
            beta: self.beta,
            k: self.k,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("local_response_norm_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups_x = (width as u32).div_ceil(8);
        let workgroups_y = (height as u32).div_ceil(8);
        let workgroups_z = ((batch_size * channels) as u32).div_ceil(256);

        ComputeDispatch::new(device, "LocalResponseNorm")
            .shader(
                include_str!("../shaders/norm/local_response_norm_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply local response normalization
    /// # Arguments
    /// - `size`: Neighborhood size
    /// - `alpha`: Scaling parameter (typically 1e-4)
    /// - `beta`: Exponent (typically 0.75)
    /// - `k`: Bias (typically 1.0 or 2.0)
    /// # Errors
    /// Returns [`Err`] if input is not 4D, size is zero, buffer allocation fails, GPU dispatch
    /// fails, or the device is lost.
    pub fn local_response_norm(self, size: usize, alpha: f32, beta: f32, k: f32) -> Result<Self> {
        LocalResponseNorm::new(self, size, alpha, beta, k)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_local_response_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();

        let output = input.local_response_norm(5, 1e-4, 0.75, 1.0).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 3, 4, 4]);
        assert_eq!(result.len(), 48);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}
