// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NORM_CDF` — Normal distribution CDF and PDF — Pure WGSL via ComputeDispatch builder.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Normal distribution CDF Φ(x) and PDF φ(x)
pub struct NormCdf {
    input: Tensor,
    mu: f32,
    sigma: f32,
    compute_pdf: bool,
}

impl NormCdf {
    /// Create standard normal CDF operation (μ=0, σ=1)
    #[must_use]
    pub fn standard_cdf(input: Tensor) -> Self {
        Self {
            input,
            mu: 0.0,
            sigma: 1.0,
            compute_pdf: false,
        }
    }

    /// Create standard normal PDF operation (μ=0, σ=1)
    #[must_use]
    pub fn standard_pdf(input: Tensor) -> Self {
        Self {
            input,
            mu: 0.0,
            sigma: 1.0,
            compute_pdf: true,
        }
    }

    /// Create general normal CDF operation with custom μ, σ
    #[must_use]
    pub fn cdf(input: Tensor, mu: f32, sigma: f32) -> Self {
        Self {
            input,
            mu,
            sigma,
            compute_pdf: false,
        }
    }

    /// Create general normal PDF operation with custom μ, σ
    #[must_use]
    pub fn pdf(input: Tensor, mu: f32, sigma: f32) -> Self {
        Self {
            input,
            mu,
            sigma,
            compute_pdf: true,
        }
    }

    /// Execute normal CDF or PDF on the input tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
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
            mode: u32, // 0 = CDF, 1 = PDF
        }

        let params = Params {
            size: size as u32,
            mu: self.mu,
            sigma: self.sigma,
            mode: u32::from(self.compute_pdf),
        };
        let params_buffer = device.create_uniform_buffer("NormCdf Params", &params);

        ComputeDispatch::new(device, "norm_cdf")
            .shader(
                include_str!("../shaders/special/norm_cdf_f64.wgsl"),
                "main",
            )
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
    /// Compute standard normal CDF Φ(x) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
    pub fn norm_cdf(self) -> Result<Self> {
        NormCdf::standard_cdf(self).execute()
    }

    /// Compute standard normal PDF φ(x) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
    pub fn norm_pdf(self) -> Result<Self> {
        NormCdf::standard_pdf(self).execute()
    }

    /// Compute general normal CDF Φ(x; μ, σ) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
    pub fn norm_cdf_params(self, mu: f32, sigma: f32) -> Result<Self> {
        NormCdf::cdf(self, mu, sigma).execute()
    }

    /// Compute general normal PDF φ(x; μ, σ) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, the
    /// device is lost, or compute submission fails.
    pub fn norm_pdf_params(self, mu: f32, sigma: f32) -> Result<Self> {
        NormCdf::pdf(self, mu, sigma).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_norm_cdf_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![0.0], vec![1], device);
        let output = input.norm_cdf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 0.5).abs() < 0.001,
            "Φ(0) = {}, expected 0.5",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_norm_cdf_critical() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![-1.96, 1.96], vec![2], device);
        let output = input.norm_cdf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 0.025).abs() < 0.01,
            "Φ(-1.96) = {}, expected ~0.025",
            result[0]
        );
        assert!(
            (result[1] - 0.975).abs() < 0.01,
            "Φ(1.96) = {}, expected ~0.975",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_norm_pdf_peak() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![0.0], vec![1], device);
        let output = input.norm_pdf().unwrap();
        let result = output.to_vec().unwrap();
        let expected = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
        assert!(
            (result[0] - expected).abs() < 0.001,
            "φ(0) = {}, expected {}",
            result[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_norm_cdf_general() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![5.0], vec![1], device);
        let output = input.norm_cdf_params(5.0, 2.0).unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 0.5).abs() < 0.001,
            "Φ(5; 5, 2) = {}, expected 0.5",
            result[0]
        );
    }
}
