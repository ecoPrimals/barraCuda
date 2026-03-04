// SPDX-License-Identifier: AGPL-3.0-or-later
//! RAWR Weighted Mean (f64) — groundSpring V10/V54
//!
//! Reliability-Averaged Weighted Resampling: weighted mean with bootstrap confidence intervals.

use std::sync::Arc;

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};

/// WGSL shader for RAWR weighted mean.
pub const WGSL_RAWR_WEIGHTED_MEAN_F64: &str =
    include_str!("../shaders/special/rawr_weighted_mean_f64.wgsl");

/// Result of RAWR weighted mean with bootstrap confidence intervals.
#[derive(Debug, Clone)]
pub struct RawrResult {
    /// Point estimate (weighted mean of original data)
    pub mean: f64,
    /// Lower bound of confidence interval
    pub ci_lower: f64,
    /// Upper bound of confidence interval
    pub ci_upper: f64,
    /// Bootstrap means (one per resample)
    pub bootstrap_means: Vec<f64>,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuParams {
    n: u32,
    n_resamples: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Simple LCG for bootstrap index generation.
fn generate_resample_indices(n: usize, n_resamples: usize, seed: u64) -> Vec<u32> {
    let mut indices = Vec::with_capacity(n_resamples * n);
    let mut state = seed;
    for _ in 0..n_resamples {
        for _ in 0..n {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let idx = (state as usize) % n;
            indices.push(idx as u32);
        }
    }
    indices
}

/// GPU-backed RAWR weighted mean.
pub struct RawrWeightedMeanGpu;

impl RawrWeightedMeanGpu {
    /// Execute RAWR weighted mean on GPU.
    ///
    /// Generates bootstrap resamples, computes weighted mean per resample on GPU,
    /// then computes confidence intervals from sorted bootstrap means on CPU.
    pub fn execute(
        device: Arc<WgpuDevice>,
        data: &[f64],
        weights: &[f64],
        n_resamples: usize,
        seed: u64,
        confidence: f64,
    ) -> Result<RawrResult> {
        if data.len() != weights.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "data and weights must have same length: {} vs {}",
                    data.len(),
                    weights.len()
                ),
            });
        }
        let n = data.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "data cannot be empty".to_string(),
            });
        }
        if n_resamples == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "n_resamples must be > 0".to_string(),
            });
        }
        if !(0.0..1.0).contains(&confidence) {
            return Err(BarracudaError::InvalidInput {
                message: format!("confidence must be in (0, 1), got {}", confidence),
            });
        }

        let sum_wx: f64 = data.iter().zip(weights.iter()).map(|(d, w)| d * w).sum();
        let sum_w: f64 = weights.iter().sum();
        let mean = if sum_w > 0.0 { sum_wx / sum_w } else { 0.0 };

        let indices = generate_resample_indices(n, n_resamples, seed);

        let data_buf = device.create_buffer_f64_init("rawr:data", data);
        let weights_buf = device.create_buffer_f64_init("rawr:weights", weights);
        let indices_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rawr:indices"),
                contents: bytemuck::cast_slice(&indices),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = device.create_buffer_f64(n_resamples)?;

        let params = GpuParams {
            n: n as u32,
            n_resamples: n_resamples as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = device.create_uniform_buffer("rawr:params", &params);

        let wg_count = n_resamples.div_ceil(WORKGROUP_SIZE_1D as usize) as u32;

        ComputeDispatch::new(&device, "rawr_weighted_mean_f64")
            .shader(WGSL_RAWR_WEIGHTED_MEAN_F64, "main")
            .f64()
            .storage_read(0, &data_buf)
            .storage_read(1, &weights_buf)
            .storage_read(2, &indices_buf)
            .storage_rw(3, &out_buf)
            .uniform(4, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit();

        let mut bootstrap_means = device.read_f64_buffer(&out_buf, n_resamples)?;

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence;
        let lo_idx = ((alpha / 2.0) * n_resamples as f64).floor() as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n_resamples as f64).ceil() as usize;
        let lo_idx = lo_idx.min(n_resamples.saturating_sub(1));
        let hi_idx = hi_idx.min(n_resamples.saturating_sub(1));

        let ci_lower = bootstrap_means[lo_idx];
        let ci_upper = bootstrap_means[hi_idx];

        Ok(RawrResult {
            mean,
            ci_lower,
            ci_upper,
            bootstrap_means,
        })
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_rawr_construction() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0];

        let result = RawrWeightedMeanGpu::execute(device, &data, &weights, 100, 42, 0.95).unwrap();
        assert_eq!(result.bootstrap_means.len(), 100);
        assert!((result.mean - 3.0).abs() < 1e-10);
        assert!(result.ci_lower < result.mean);
        assert!(result.ci_upper > result.mean);
    }

    #[tokio::test]
    async fn test_rawr_dimension_check() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 1.0];

        let result = RawrWeightedMeanGpu::execute(device, &data, &weights, 10, 42, 0.95);
        assert!(result.is_err());
    }
}
