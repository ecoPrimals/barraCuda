// SPDX-License-Identifier: AGPL-3.0-or-later
//! HERMITE - Physicist's Hermite polynomials - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Hermite polynomial evaluator Hₙ(x)
pub struct Hermite {
    input: Tensor,
    n: u32,
}

impl Hermite {
    /// Create new Hermite polynomial operation for order n
    #[must_use]
    pub fn new(input: Tensor, n: u32) -> Self {
        Self { input, n }
    }

    /// Execute Hermite polynomial evaluation on the input tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
        }

        let params = Params {
            size: size as u32,
            n: self.n,
        };
        let params_buffer = device.create_uniform_buffer("Hermite Params", &params);
        let input_buffer = self.input.buffer();

        ComputeDispatch::new(device, "Hermite")
            .shader(include_str!("../shaders/special/hermite.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Hermite polynomial Hₙ(x) for each element
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn hermite(self, n: u32) -> Result<Self> {
        Hermite::new(self, n).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hermite_h0() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let input = Tensor::new(data, vec![5], device);
        let output = input.hermite(0).unwrap();
        let result = output.to_vec().unwrap();
        // H₀(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-5, "H₀ should be 1, got {v}");
        }
    }

    #[tokio::test]
    async fn test_hermite_h1() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.hermite(1).unwrap();
        let result = output.to_vec().unwrap();
        // H₁(x) = 2x
        for (i, &v) in result.iter().enumerate() {
            let expected = 2.0 * data[i];
            assert!(
                (v - expected).abs() < 1e-5,
                "H₁({}) = {}, expected {}",
                data[i],
                v,
                expected
            );
        }
    }

    #[tokio::test]
    async fn test_hermite_h2() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.hermite(2).unwrap();
        let result = output.to_vec().unwrap();
        // H₂(x) = 4x² - 2
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = (4.0 * x).mul_add(x, -2.0);
            assert!(
                (v - expected).abs() < 1e-4,
                "H₂({x}) = {v}, expected {expected}"
            );
        }
    }
}
