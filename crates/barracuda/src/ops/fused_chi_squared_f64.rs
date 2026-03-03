//! Fused Chi-Squared Test (f64) — neuralSpring V24
//!
//! Fused chi-squared test: computes observed vs expected, chi-squared statistic,
//! and p-value in a single GPU dispatch.

use std::sync::Arc;

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::special::chi_squared_sf;
use bytemuck::{Pod, Zeroable};

/// WGSL shader for fused chi-squared test.
pub const WGSL_FUSED_CHI_SQUARED_F64: &str =
    include_str!("../shaders/special/fused_chi_squared_f64.wgsl");

/// Result of a chi-squared goodness-of-fit test.
#[derive(Debug, Clone)]
pub struct ChiSquaredTestResult {
    /// Chi-squared statistic
    pub statistic: f64,
    /// P-value (probability of observing this or more extreme under null)
    pub p_value: f64,
    /// Degrees of freedom
    pub degrees_of_freedom: usize,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// GPU-backed fused chi-squared test.
pub struct FusedChiSquaredGpu;

impl FusedChiSquaredGpu {
    /// Execute fused chi-squared test on GPU.
    ///
    /// Computes χ² = Σ (observed - expected)² / expected and p-value via
    /// incomplete gamma (survival function).
    pub fn execute(
        device: Arc<WgpuDevice>,
        observed: &[f64],
        expected: &[f64],
    ) -> Result<ChiSquaredTestResult> {
        if observed.len() != expected.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "observed and expected must have same length: {} vs {}",
                    observed.len(),
                    expected.len()
                ),
            });
        }
        let n = observed.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "observed and expected cannot be empty".to_string(),
            });
        }
        for (i, &e) in expected.iter().enumerate() {
            if e <= 0.0 {
                return Err(BarracudaError::InvalidInput {
                    message: format!("expected[{}] must be positive, got {}", i, e),
                });
            }
        }

        let wg_size = 256u32;
        let n_workgroups = n.div_ceil(wg_size as usize) as u32;

        let observed_buf = device.create_buffer_f64_init("fused_chi2:observed", observed);
        let expected_buf = device.create_buffer_f64_init("fused_chi2:expected", expected);
        let partial_buf = device.create_buffer_f64(n_workgroups as usize)?;

        let params = GpuParams {
            n: n as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buf = device.create_uniform_buffer("fused_chi2:params", &params);

        ComputeDispatch::new(&device, "fused_chi_squared_f64")
            .shader(WGSL_FUSED_CHI_SQUARED_F64, "main")
            .f64()
            .storage_read(0, &observed_buf)
            .storage_read(1, &expected_buf)
            .storage_rw(2, &partial_buf)
            .uniform(3, &params_buf)
            .dispatch(n_workgroups, 1, 1)
            .submit();

        let partials = device.read_f64_buffer(&partial_buf, n_workgroups as usize)?;
        let statistic = partials.iter().sum::<f64>();

        let df = n - 1;
        let p_value = chi_squared_sf(statistic, df as f64)?;

        Ok(ChiSquaredTestResult {
            statistic,
            p_value,
            degrees_of_freedom: df,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_fused_chi_squared_construction() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let observed = vec![10.0, 10.0, 10.0, 10.0, 10.0, 10.0];
        let expected = vec![10.0; 6];

        let result = FusedChiSquaredGpu::execute(device, &observed, &expected).unwrap();
        assert_eq!(result.degrees_of_freedom, 5);
        assert!((result.statistic - 0.0).abs() < 1e-10);
        assert!(result.p_value > 0.99);
    }

    #[tokio::test]
    async fn test_fused_chi_squared_dimension_check() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let observed = vec![10.0, 10.0];
        let expected = vec![10.0, 10.0, 10.0];

        let result = FusedChiSquaredGpu::execute(device, &observed, &expected);
        assert!(result.is_err());
    }
}
