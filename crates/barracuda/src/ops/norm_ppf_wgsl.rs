// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NORM_PPF` — Inverse Normal CDF (Probit) — Pure WGSL via ComputeDispatch builder.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Inverse Normal CDF (Percent Point Function / Probit)
pub struct NormPpf {
    input: Tensor,
    mu: f32,
    sigma: f32,
}

impl NormPpf {
    /// Create standard normal inverse CDF (μ=0, σ=1)
    #[must_use]
    pub fn standard(input: Tensor) -> Self {
        Self {
            input,
            mu: 0.0,
            sigma: 1.0,
        }
    }

    /// Create general normal inverse CDF with custom μ, σ
    #[must_use]
    pub fn general(input: Tensor, mu: f32, sigma: f32) -> Self {
        Self { input, mu, sigma }
    }

    /// Execute inverse normal CDF (probit) on the input tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();
        let input_buffer = self.input.buffer();
        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            mu: f32,
            sigma: f32,
        }

        let params = Params {
            size: size as u32,
            mu: self.mu,
            sigma: self.sigma,
        };
        let params_buffer = device.create_uniform_buffer("NormPpf Params", &params);

        ComputeDispatch::new(device, "norm_ppf")
            .shader(include_str!("../shaders/special/norm_ppf_f64.wgsl"), "main")
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
    /// Compute standard normal inverse CDF Φ⁻¹(p) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn norm_ppf(self) -> Result<Self> {
        NormPpf::standard(self).execute()
    }

    /// Compute general normal inverse CDF with custom μ, σ
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn norm_ppf_params(self, mu: f32, sigma: f32) -> Result<Self> {
        NormPpf::general(self, mu, sigma).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_norm_ppf_median() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![0.5], vec![1], device);
        let output = input.norm_ppf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            result[0].abs() < 0.001,
            "Φ⁻¹(0.5) = {}, expected 0",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_norm_ppf_quartiles() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![0.25, 0.75], vec![2], device);
        let output = input.norm_ppf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - (-0.6745)).abs() < 0.01,
            "Φ⁻¹(0.25) = {}, expected -0.6745",
            result[0]
        );
        assert!(
            (result[1] - 0.6745).abs() < 0.01,
            "Φ⁻¹(0.75) = {}, expected 0.6745",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_norm_ppf_critical() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![0.025, 0.975], vec![2], device);
        let output = input.norm_ppf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - (-1.96)).abs() < 0.02,
            "Φ⁻¹(0.025) = {}, expected -1.96",
            result[0]
        );
        assert!(
            (result[1] - 1.96).abs() < 0.02,
            "Φ⁻¹(0.975) = {}, expected 1.96",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_norm_ppf_general() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![0.5], vec![1], device);
        let output = input.norm_ppf_params(10.0, 2.0).unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 10.0).abs() < 0.01,
            "Φ⁻¹(0.5; 10, 2) = {}, expected 10",
            result[0]
        );
    }
}
