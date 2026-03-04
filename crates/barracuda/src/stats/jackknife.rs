// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leave-one-out jackknife estimator.
//!
//! Provenance: groundSpring `jackknife.rs` -> toadStool absorption (S70).

#[cfg(feature = "gpu")]
use crate::device::capabilities::WORKGROUP_SIZE_1D;
#[cfg(feature = "gpu")]
use crate::device::compute_pipeline::ComputeDispatch;
#[cfg(feature = "gpu")]
use crate::device::WgpuDevice;
#[cfg(feature = "gpu")]
use crate::error::Result;
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
const SHADER_JACKKNIFE: &str = include_str!("../shaders/stats/jackknife_mean_f64.wgsl");

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct JackknifeParams {
    n: u32,
    _pad: u32,
}

#[cfg(feature = "gpu")]
/// GPU-parallel leave-one-out jackknife for the mean.
pub struct JackknifeMeanGpu {
    device: Arc<WgpuDevice>,
}

#[cfg(feature = "gpu")]
impl JackknifeMeanGpu {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch GPU to compute leave-out means, then compute variance on CPU.
    pub fn dispatch(&self, data: &[f64]) -> Result<JackknifeResult> {
        let n = data.len();
        if n < 2 {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "jackknife requires at least 2 observations".to_string(),
            });
        }

        let full_sum: f64 = data.iter().sum();
        let full_mean = full_sum / (n as f64);

        let params = JackknifeParams {
            n: n as u32,
            _pad: 0,
        };
        let params_buf = self
            .device
            .create_uniform_buffer("jackknife_mean:params", &params);

        let data_buf = self
            .device
            .create_buffer_f64_init("jackknife_mean:data", data);
        let leave_means_buf = self.device.create_buffer_f64(n)?;
        let full_sum_buf = self
            .device
            .create_buffer_f64_init("jackknife_mean:full_sum", &[full_sum]);

        let wg_count = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "jackknife_mean")
            .shader(SHADER_JACKKNIFE, "main")
            .f64()
            .storage_read(0, &data_buf)
            .storage_rw(1, &leave_means_buf)
            .uniform(2, &params_buf)
            .storage_read(3, &full_sum_buf)
            .dispatch(wg_count, 1, 1)
            .submit();

        let leave_means = self.device.read_f64_buffer(&leave_means_buf, n)?;

        let n_f = n as f64;
        let jk_grand_mean: f64 = leave_means.iter().sum::<f64>() / n_f;
        let jk_var = (n_f - 1.0) / n_f
            * leave_means
                .iter()
                .map(|&m| (m - jk_grand_mean).powi(2))
                .sum::<f64>();

        Ok(JackknifeResult {
            estimate: full_mean,
            variance: jk_var,
            std_error: jk_var.sqrt(),
        })
    }
}

/// Result of a jackknife estimate.
#[derive(Debug, Clone, Copy)]
pub struct JackknifeResult {
    /// Jackknife estimate of the statistic.
    pub estimate: f64,
    /// Jackknife variance of the estimator.
    pub variance: f64,
    /// Standard error (sqrt of variance).
    pub std_error: f64,
}

/// Leave-one-out jackknife for the mean.
///
/// Returns `None` if fewer than 2 observations.
///
/// # Complexity
/// O(n) time, O(n) space for leave-one-out means.
#[must_use]
pub fn jackknife_mean_variance(data: &[f64]) -> Option<JackknifeResult> {
    let n = data.len();
    if n < 2 {
        return None;
    }

    let n_f = n as f64;
    let full_sum: f64 = data.iter().sum();
    let full_mean = full_sum / n_f;

    let mut jk_mean_sum = 0.0;
    let mut jk_means = Vec::with_capacity(n);

    for &d in data {
        let leave_mean = (full_sum - d) / (n_f - 1.0);
        jk_means.push(leave_mean);
        jk_mean_sum += leave_mean;
    }

    let jk_grand_mean = jk_mean_sum / n_f;
    let jk_var = (n_f - 1.0) / n_f
        * jk_means
            .iter()
            .map(|&m| (m - jk_grand_mean).powi(2))
            .sum::<f64>();

    Some(JackknifeResult {
        estimate: full_mean,
        variance: jk_var,
        std_error: jk_var.sqrt(),
    })
}

/// Generalized jackknife for an arbitrary statistic.
///
/// `statistic` is called n+1 times: once for the full dataset, then n times
/// with each observation removed.
///
/// Returns `None` if fewer than 2 observations.
#[must_use]
pub fn jackknife<F>(data: &[f64], statistic: F) -> Option<JackknifeResult>
where
    F: Fn(&[f64]) -> f64,
{
    let n = data.len();
    if n < 2 {
        return None;
    }

    let n_f = n as f64;
    let full_stat = statistic(data);

    let mut leave_out = Vec::with_capacity(n - 1);
    let mut pseudovalues = Vec::with_capacity(n);

    for i in 0..n {
        leave_out.clear();
        leave_out.extend_from_slice(&data[..i]);
        leave_out.extend_from_slice(&data[i + 1..]);
        let theta_i = statistic(&leave_out);
        pseudovalues.push(n_f * full_stat - (n_f - 1.0) * theta_i);
    }

    let mean_pseudo: f64 = pseudovalues.iter().sum::<f64>() / n_f;
    let var = pseudovalues
        .iter()
        .map(|&p| (p - mean_pseudo).powi(2))
        .sum::<f64>()
        / (n_f * (n_f - 1.0));

    Some(JackknifeResult {
        estimate: mean_pseudo,
        variance: var,
        std_error: var.sqrt(),
    })
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jackknife_mean_basic() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = jackknife_mean_variance(&data).unwrap();
        assert!((result.estimate - 3.0).abs() < 1e-12);
        assert!(result.variance >= 0.0);
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn test_jackknife_mean_two_elements() {
        let data = [10.0, 20.0];
        let result = jackknife_mean_variance(&data).unwrap();
        assert!((result.estimate - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_jackknife_mean_too_few() {
        assert!(jackknife_mean_variance(&[]).is_none());
        assert!(jackknife_mean_variance(&[1.0]).is_none());
    }

    #[test]
    fn test_jackknife_constant() {
        let data = [5.0; 10];
        let result = jackknife_mean_variance(&data).unwrap();
        assert!((result.estimate - 5.0).abs() < 1e-12);
        assert!(result.variance < 1e-20);
    }

    #[test]
    fn test_jackknife_generalized() {
        let data = [2.0, 4.0, 6.0, 8.0];
        let result = jackknife(&data, |d| d.iter().sum::<f64>() / d.len() as f64).unwrap();
        assert!((result.estimate - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_jackknife_generalized_too_few_elements() {
        assert!(jackknife(&[], |d| d.iter().sum::<f64>() / d.len().max(1) as f64).is_none());
        assert!(jackknife(&[1.0], |d| d.iter().sum::<f64>()).is_none());
    }

    #[test]
    fn test_jackknife_generalized_variance_statistic() {
        fn variance(d: &[f64]) -> f64 {
            let n = d.len() as f64;
            if n < 2.0 {
                return 0.0;
            }
            let mean: f64 = d.iter().sum::<f64>() / n;
            d.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
        }
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = jackknife(&data, variance).unwrap();
        assert!(result.variance >= 0.0);
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn test_jackknife_mean_variance_large_dataset() {
        // Use data with roughly constant variance (uniform [0,1]) so std_error ∝ 1/√n
        let n = 150;
        let data: Vec<f64> = (0..n).map(|i| (i % 10) as f64 / 10.0).collect();
        let result = jackknife_mean_variance(&data).unwrap();
        let expected_mean: f64 = data.iter().sum::<f64>() / n as f64;
        assert!((result.estimate - expected_mean).abs() < 1e-10);
        let small_result = jackknife_mean_variance(&data[..20]).unwrap();
        assert!(
            result.std_error < small_result.std_error,
            "std_error should decrease with n (large n={} vs small n=20)",
            n
        );
    }

    #[test]
    fn test_jackknife_mean_variance_negative_values() {
        let data = [-5.0, -2.0, 0.0, 3.0, 4.0];
        let result = jackknife_mean_variance(&data).unwrap();
        assert!((result.estimate - 0.0).abs() < 1e-12);
        assert!(result.variance >= 0.0);
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_jackknife_gpu_dispatch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let cpu_result = jackknife_mean_variance(&data).unwrap();
        let gpu = match super::JackknifeMeanGpu::new(device) {
            Ok(g) => g,
            Err(e) if e.is_device_lost() => return,
            Err(e) => panic!("unexpected: {e}"),
        };
        let gpu_result = match gpu.dispatch(&data) {
            Ok(r) => r,
            Err(e) if e.is_device_lost() => return,
            Err(e) => panic!("unexpected: {e}"),
        };
        assert!((cpu_result.estimate - gpu_result.estimate).abs() < 1e-10);
        assert!((cpu_result.variance - gpu_result.variance).abs() < 1e-10);
    }
}
