// SPDX-License-Identifier: AGPL-3.0-or-later
//! Softplus - Smooth approximation of `ReLU` - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its beta parameter
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Softplus operation - Smooth approximation of `ReLU`
///
/// Applies Softplus(x) = (1/beta) * log(1 + exp(beta * x))
pub struct Softplus {
    input: Tensor,
    beta: f32,
}

impl Softplus {
    /// Create a new Softplus operation
    #[must_use]
    pub fn new(input: Tensor, beta: f32) -> Self {
        Self { input, beta }
    }

    /// Execute the softplus operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            beta: f32,
        }

        let params = Params {
            size: size as u32,
            beta: self.beta,
        };

        let params_buffer = device.create_uniform_buffer("Softplus Params", &params);

        ComputeDispatch::new(device, "Softplus")
            .shader(include_str!("../shaders/activation/softplus_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Apply Softplus activation
    /// # Arguments
    /// * `beta` - Smoothness parameter (typically 1.0)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn softplus_wgsl(self, beta: f32) -> Result<Self> {
        Softplus::new(self, beta).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_softplus_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-2.0, 0.0, 2.0];
        let input = Tensor::new(data, vec![3], device);

        let output = input.softplus_wgsl(1.0).unwrap();

        assert_eq!(output.shape(), &[3]);
        let result = output.to_vec().unwrap();

        // Softplus should be close to 0 for very negative values
        assert!(result[0] < 0.2);
        // Softplus(0) ≈ ln(2) ≈ 0.693
        assert!((result[1] - 0.693).abs() < 0.01);
        // Softplus should be close to x for large positive values
        assert!((result[2] - 2.0).abs() < 0.2);
    }

    #[tokio::test]
    async fn test_softplus_beta() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0];
        let input = Tensor::new(data, vec![2], device);

        let output = input.softplus_wgsl(2.0).unwrap();

        let result = output.to_vec().unwrap();
        // With beta=2, the function is steeper
        assert!(result[0] > 0.3); // softplus(0, beta=2)
        assert!(result[1] > 0.8); // softplus(1, beta=2)
    }
}
