// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PReLU` - Parametric Rectified Linear Unit - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its alpha parameter
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// `PReLU` operation - Parametric Rectified Linear Unit
pub struct PReLU {
    input: Tensor,
    alpha: f32,
}

impl PReLU {
    /// Create a new `PReLU` operation
    #[must_use]
    pub fn new(input: Tensor, alpha: f32) -> Self {
        Self { input, alpha }
    }

    /// Execute the `PReLU` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            alpha: f32,
        }

        let params = Params {
            size: size as u32,
            alpha: self.alpha,
        };

        let params_buffer = device.create_uniform_buffer("PReLU Params", &params);

        ComputeDispatch::new(device, "PReLU")
            .shader(include_str!("../shaders/activation/prelu_f64.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(
            output_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply Parametric `ReLU` activation
    ///
    /// # Arguments
    ///
    /// * `alpha` - Slope for negative values (typically 0.01 to 0.3)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn prelu_wgsl(self, alpha: f32) -> Result<Self> {
        PReLU::new(self, alpha).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prelu_positive() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0];
        let input = Tensor::new(data, vec![3], device);

        let output = input.prelu_wgsl(0.1).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[tokio::test]
    async fn test_prelu_negative() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -2.0, -3.0];
        let input = Tensor::new(data, vec![3], device);

        let output = input.prelu_wgsl(0.25).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], -0.25);
        assert_eq!(result[1], -0.5);
        assert_eq!(result[2], -0.75);
    }

    #[tokio::test]
    async fn test_prelu_mixed() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-2.0, 0.0, 2.0];
        let input = Tensor::new(data, vec![3], device);

        let output = input.prelu_wgsl(0.1).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], -0.2);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 2.0);
    }
}
