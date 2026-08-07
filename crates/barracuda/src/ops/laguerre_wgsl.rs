// SPDX-License-Identifier: AGPL-3.0-or-later
//! LAGUERRE - Generalized Laguerre polynomials - Pure WGSL
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

/// Generalized Laguerre polynomial evaluator `L_n^(α)(x)`
pub struct Laguerre {
    input: Tensor,
    n: u32,
    alpha: f32,
}

impl Laguerre {
    /// Create new Laguerre polynomial operation for degree n and parameter α
    #[must_use]
    pub fn new(input: Tensor, n: u32, alpha: f32) -> Self {
        Self { input, n, alpha }
    }

    /// Create simple Laguerre polynomial `L_n(x)` = `L_n^(0)(x)`
    #[must_use]
    pub fn simple(input: Tensor, n: u32) -> Self {
        Self::new(input, n, 0.0)
    }

    /// Execute Laguerre polynomial evaluation on the input tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
            alpha: f32,
        }

        let params = Params {
            size: size as u32,
            n: self.n,
            alpha: self.alpha,
        };
        let params_buffer = device.create_uniform_buffer("Laguerre Params", &params);

        ComputeDispatch::new(device, "Laguerre")
            .shader(include_str!("../shaders/special/laguerre.wgsl"), "main")
            .storage_read(0, self.input.buffer())
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
    /// Compute generalized Laguerre polynomial `L_n^(α)(x)` for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn laguerre(self, n: u32, alpha: f32) -> Result<Self> {
        Laguerre::new(self, n, alpha).execute()
    }

    /// Compute simple Laguerre polynomial `L_n(x)` = `L_n^(0)(x)` for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn laguerre_simple(self, n: u32) -> Result<Self> {
        Laguerre::simple(self, n).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_laguerre_l0() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0, 2.0, 5.0];
        let input = Tensor::new(data, vec![4], device);
        let output = input.laguerre_simple(0).unwrap();
        let result = output.to_vec().unwrap();
        // L_0(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-5, "L_0 should be 1, got {v}");
        }
    }

    #[tokio::test]
    async fn test_laguerre_l1() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0, 3.0];
        let input = Tensor::new(data.clone(), vec![3], device);
        let output = input.laguerre_simple(1).unwrap();
        let result = output.to_vec().unwrap();
        // L_1(x) = 1 - x
        let expected = [1.0, 0.0, -2.0];
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "L_1({}) = {}, expected {}",
                data[i],
                v,
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_laguerre_l2() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.0, 1.0, 2.0];
        let input = Tensor::new(data.clone(), vec![3], device);
        let output = input.laguerre_simple(2).unwrap();
        let result = output.to_vec().unwrap();
        // L_2(x) = (x² - 4x + 2) / 2
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = f32::midpoint(4.0f32.mul_add(-x, x * x), 2.0);
            assert!(
                (v - expected).abs() < 1e-4,
                "L_2({x}) = {v}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_laguerre_generalized() {
        let device = crate::device::test_pool::get_test_device().await;
        // L_1^(1)(x) = 2 - x
        let data = vec![0.0, 2.0];
        let input = Tensor::new(data, vec![2], device);
        let output = input.laguerre(1, 1.0).unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 2.0).abs() < 1e-5,
            "L_1^(1)(0) = {}, expected 2",
            result[0]
        );
        assert!(
            (result[1] - 0.0).abs() < 1e-5,
            "L_1^(1)(2) = {}, expected 0",
            result[1]
        );
    }
}
