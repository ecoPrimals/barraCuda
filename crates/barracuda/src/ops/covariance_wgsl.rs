// SPDX-License-Identifier: AGPL-3.0-only
//! Covariance — GPU-Accelerated via WGSL
//!
//! Computes covariance: Cov(X,Y) = E[(X-μx)(Y-μy)]
//!
//! **Use cases**:
//! - Portfolio theory (wetSpring)
//! - PCA preprocessing (all springs)
//! - Kalman filters (airSpring sensor fusion)
//!
//! **Note**: f32 precision. For f64, use manual computation with `weighted_dot_f64`.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for covariance shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CovarianceParams {
    size: u32,
    num_pairs: u32,
    stride: u32,
    ddof: u32, // Delta degrees of freedom (0=population, 1=sample)
}

/// GPU-accelerated covariance computation
pub struct Covariance {
    device: Arc<WgpuDevice>,
}

impl Covariance {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/covariance.wgsl")
    }

    /// Create a new Covariance orchestrator
    /// # Errors
    /// Never returns an error; always returns `Ok`.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute sample covariance between two vectors (ddof=1)
    /// # Arguments
    /// * `x` - First vector (f32)
    /// * `y` - Second vector (f32)
    /// # Returns
    /// Sample covariance
    /// # Errors
    /// Returns [`Err`] if `x.len() != y.len()` (invalid input), if buffer
    /// allocation fails, or if buffer mapping fails (e.g. device lost,
    /// mapping channel closed).
    pub fn covariance(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        self.covariance_with_ddof(x, y, 1)
    }

    /// Compute population covariance (ddof=0)
    /// # Errors
    /// Returns [`Err`] if `x.len() != y.len()` or `n <= ddof` (invalid input),
    /// if buffer allocation fails, or if buffer mapping fails (e.g. device lost,
    /// mapping channel closed).
    pub fn population_covariance(&self, x: &[f32], y: &[f32]) -> Result<f32> {
        self.covariance_with_ddof(x, y, 0)
    }

    /// Compute covariance with specified degrees of freedom
    /// # Errors
    /// Returns [`Err`] if `x.len() != y.len()` or `n <= ddof` (invalid input),
    /// if buffer allocation fails, or if buffer mapping fails (e.g. device lost,
    /// mapping channel closed).
    pub fn covariance_with_ddof(&self, x: &[f32], y: &[f32], ddof: u32) -> Result<f32> {
        let n = x.len();
        if y.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!("Vector lengths must match: x={}, y={}", n, y.len()),
            });
        }

        if n <= ddof as usize {
            return Err(BarracudaError::InvalidInput {
                message: format!("Need more than {ddof} elements for ddof={ddof}"),
            });
        }

        self.covariance_gpu(x, y, ddof)
    }

    /// Compute sample variance of a single vector (ddof=1)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails or if buffer mapping fails
    /// (e.g. device lost, mapping channel closed).
    pub fn variance(&self, x: &[f32]) -> Result<f32> {
        self.covariance_with_ddof(x, x, 1)
    }

    /// Compute population variance (ddof=0)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails or if buffer mapping fails
    /// (e.g. device lost, mapping channel closed).
    pub fn population_variance(&self, x: &[f32]) -> Result<f32> {
        self.covariance_with_ddof(x, x, 0)
    }

    /// Compute standard deviation (sqrt of variance)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails or if buffer mapping fails
    /// (e.g. device lost, mapping channel closed).
    pub fn std(&self, x: &[f32]) -> Result<f32> {
        Ok(self.variance(x)?.sqrt())
    }

    /// CPU reference implementation
    #[expect(
        dead_code,
        reason = "CPU reference implementation for GPU parity validation"
    )]
    fn covariance_cpu(&self, x: &[f32], y: &[f32], ddof: u32) -> f32 {
        let n = x.len() as f32;
        let mean_x: f32 = x.iter().sum::<f32>() / n;
        let mean_y: f32 = y.iter().sum::<f32>() / n;

        let cov_sum: f32 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        cov_sum / (n - ddof as f32)
    }

    fn covariance_gpu(&self, x: &[f32], y: &[f32], ddof: u32) -> Result<f32> {
        let n = x.len();

        let x_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Covariance X"),
                contents: bytemuck::cast_slice(x),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let y_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Covariance Y"),
                contents: bytemuck::cast_slice(y),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = self.device.create_buffer_f32(1)?;

        let params = CovarianceParams {
            size: n as u32,
            num_pairs: 1,
            stride: n as u32,
            ddof,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("Covariance Params", &params);

        ComputeDispatch::new(&self.device, "Covariance")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, &x_buf)
            .storage_read(1, &y_buf)
            .storage_rw(2, &output_buf)
            .uniform(3, &params_buf)
            .dispatch(1, 1, 1)
            .submit()?;

        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Covariance Staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Covariance Copy"),
            });
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, 4);
        self.device.submit_commands(Some(encoder.finish()));

        let result_vec: Vec<f32> = self.device.map_staging_buffer(&staging, 1)?;
        Ok(result_vec[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Arc<crate::device::WgpuDevice> {
        crate::device::test_pool::get_test_device_sync()
    }

    #[test]
    fn test_variance() {
        let device = get_test_device();
        let op = Covariance::new(device).unwrap();

        let x = vec![2.0f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // Mean = 5, Var = Σ(x-5)² / (n-1) = (9+1+1+1+0+0+4+16) / 7 = 32/7 ≈ 4.57

        let var = op.variance(&x).unwrap();
        assert!((var - 4.571428).abs() < 0.01, "Expected ~4.57, got {var}");
    }

    #[test]
    fn test_covariance_positive() {
        let device = get_test_device();
        let op = Covariance::new(device).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..100).map(|i| i as f32 * 2.0).collect();

        let cov = op.covariance(&x, &y).unwrap();
        assert!(cov > 0.0, "Expected positive covariance, got {cov}");
    }

    #[test]
    fn test_covariance_negative() {
        let device = get_test_device();
        let op = Covariance::new(device).unwrap();

        let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..100).map(|i| -(i as f32)).collect();

        let cov = op.covariance(&x, &y).unwrap();
        assert!(cov < 0.0, "Expected negative covariance, got {cov}");
    }

    #[test]
    fn test_population_vs_sample() {
        let device = get_test_device();
        let op = Covariance::new(device).unwrap();

        let x: Vec<f32> = (0..10).map(|i| i as f32).collect();

        let pop_var = op.population_variance(&x).unwrap();
        let sample_var = op.variance(&x).unwrap();

        // Sample variance should be larger (n-1 denominator vs n)
        assert!(
            sample_var > pop_var,
            "Sample var ({sample_var}) should be > pop var ({pop_var})"
        );
    }
}
