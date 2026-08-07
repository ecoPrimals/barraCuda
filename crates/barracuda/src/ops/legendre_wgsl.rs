// SPDX-License-Identifier: AGPL-3.0-or-later
//! LEGENDRE - Legendre polynomials and associated functions - Pure WGSL
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

/// Legendre polynomial evaluator Pₙ(x) and associated Legendre Pₙᵐ(x)
pub struct Legendre {
    input: Tensor,
    n: u32,
    m: u32,
    is_associated: bool,
}

impl Legendre {
    /// Create new Legendre polynomial Pₙ(x)
    #[must_use]
    pub fn new(input: Tensor, n: u32) -> Self {
        Self {
            input,
            n,
            m: 0,
            is_associated: false,
        }
    }

    /// Create new associated Legendre function Pₙᵐ(x)
    #[must_use]
    pub fn associated(input: Tensor, n: u32, m: u32) -> Self {
        Self {
            input,
            n,
            m,
            is_associated: true,
        }
    }

    /// Execute Legendre polynomial evaluation on the input tensor.
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
            m: u32,
            is_assoc: u32,
        }

        let params = Params {
            size: size as u32,
            n: self.n,
            m: self.m,
            is_assoc: u32::from(self.is_associated),
        };
        let params_buffer = device.create_uniform_buffer("Legendre Params", &params);

        ComputeDispatch::new(device, "Legendre")
            .shader(include_str!("../shaders/special/legendre.wgsl"), "main")
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
    /// Compute Legendre polynomial Pₙ(x) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn legendre(self, n: u32) -> Result<Self> {
        Legendre::new(self, n).execute()
    }

    /// Compute associated Legendre function Pₙᵐ(x) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn assoc_legendre(self, n: u32, m: u32) -> Result<Self> {
        Legendre::associated(self, n, m).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_legendre_p0() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let input = Tensor::new(data, vec![5], device);
        let output = input.legendre(0).unwrap();
        let result = output.to_vec().unwrap();
        // P₀(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-5, "P₀ should be 1, got {v}");
        }
    }

    #[tokio::test]
    async fn test_legendre_p1() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.legendre(1).unwrap();
        let result = output.to_vec().unwrap();
        // P₁(x) = x
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - data[i]).abs() < 1e-5,
                "P₁({}) = {}, expected {}",
                data[i],
                v,
                data[i]
            );
        }
    }

    #[tokio::test]
    async fn test_legendre_p2() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.legendre(2).unwrap();
        let result = output.to_vec().unwrap();
        // P₂(x) = (3x² - 1) / 2
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = (3.0 * x).mul_add(x, -1.0) / 2.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "P₂({x}) = {v}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_assoc_legendre_p11() {
        let device = crate::device::test_pool::get_test_device().await;
        // P₁¹(x) = -sqrt(1 - x²) (Condon-Shortley)
        let data = vec![0.0, 0.5, -0.5];
        let input = Tensor::new(data.clone(), vec![3], device);
        let output = input.assoc_legendre(1, 1).unwrap();
        let result = output.to_vec().unwrap();
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = -(1.0 - x * x).sqrt();
            assert!(
                (v - expected).abs() < 1e-4,
                "P₁¹({x}) = {v}, expected {expected}"
            );
        }
    }
}
