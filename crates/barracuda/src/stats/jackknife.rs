// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leave-one-out jackknife estimator.
//!
//! Provenance: groundSpring `jackknife.rs` -> toadStool absorption (S70).

#[cfg(feature = "gpu")]
use crate::device::WgpuDevice;
#[cfg(feature = "gpu")]
use crate::device::capabilities::WORKGROUP_SIZE_1D;
#[cfg(feature = "gpu")]
use crate::device::compute_pipeline::ComputeDispatch;
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
    /// Creates a new GPU-accelerated jackknife mean estimator from a WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch GPU to compute leave-out means, then compute variance on CPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
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
            .submit()?;

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

    #[cfg(feature = "cpu-shader")]
    let jk_means = jackknife_leave_means_shader(data, full_sum, n);

    #[cfg(not(feature = "cpu-shader"))]
    #[expect(deprecated, reason = "fallback retained until cpu-shader is default")]
    let jk_means = jackknife_leave_means_cpu(data, full_sum, n_f);

    let jk_mean_sum: f64 = jk_means.iter().sum();
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

#[cfg(not(feature = "cpu-shader"))]
#[deprecated(
    since = "0.4.0",
    note = "use `cpu-shader` feature for WGSL-backed jackknife"
)]
fn jackknife_leave_means_cpu(data: &[f64], full_sum: f64, n_f: f64) -> Vec<f64> {
    data.iter().map(|&d| (full_sum - d) / (n_f - 1.0)).collect()
}

#[cfg(feature = "cpu-shader")]
fn jackknife_leave_means_shader(data: &[f64], full_sum: f64, n: usize) -> Vec<f64> {
    use crate::unified_hardware::{CpuShaderDispatch, ShaderBinding, ShaderDispatch};

    let wgsl = include_str!("../shaders/stats/jackknife_mean_f64.wgsl");
    let dispatcher = CpuShaderDispatch::new();

    let mut data_buf: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out_buf = vec![0u8; n * 8];

    let mut params_buf = vec![0u8; 8]; // Params { n: u32, _pad: u32 }
    params_buf[..4].copy_from_slice(&(n as u32).to_le_bytes());

    let mut sum_buf: Vec<u8> = full_sum.to_le_bytes().to_vec();

    let mut bindings = vec![
        ShaderBinding {
            group: 0,
            binding: 0,
            data: &mut data_buf,
            read_only: true,
        },
        ShaderBinding {
            group: 0,
            binding: 1,
            data: &mut out_buf,
            read_only: false,
        },
        ShaderBinding {
            group: 0,
            binding: 2,
            data: &mut params_buf,
            read_only: true,
        },
        ShaderBinding {
            group: 0,
            binding: 3,
            data: &mut sum_buf,
            read_only: true,
        },
    ];

    let workgroups = (
        (n as u32).div_ceil(crate::device::capabilities::WORKGROUP_SIZE_1D),
        1,
        1,
    );
    if dispatcher
        .dispatch_wgsl(wgsl, "main", &mut bindings, workgroups)
        .is_ok()
    {
        return out_buf
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect();
    }

    let n_f = n as f64;
    data.iter().map(|&d| (full_sum - d) / (n_f - 1.0)).collect()
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
        pseudovalues.push(n_f.mul_add(full_stat, -((n_f - 1.0) * theta_i)));
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

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;

    const JK_TOL: f64 = 1e-12;

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
            "std_error should decrease with n (large n={n} vs small n=20)"
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
        use crate::device::test_pool::test_prelude::*;
        let Some(device) = test_f64_device().await else {
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
        assert!((cpu_result.estimate - gpu_result.estimate).abs() < JK_TOL);
        assert!((cpu_result.variance - gpu_result.variance).abs() < JK_TOL);
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_jackknife_gpu_requires_at_least_two_observations() {
        use crate::device::test_pool::test_prelude::*;
        let Some(device) = test_f64_device().await else {
            return;
        };
        let gpu = match super::JackknifeMeanGpu::new(device) {
            Ok(g) => g,
            Err(e) if e.is_device_lost() => return,
            Err(e) => panic!("unexpected: {e}"),
        };
        let empty: [f64; 0] = [];
        assert!(gpu.dispatch(&empty).is_err());
        assert!(gpu.dispatch(&[1.0]).is_err());
    }

    #[test]
    fn test_jackknife_identity_statistic_matches_mean_jackknife() {
        let data = [2.0, 4.0, 6.0, 8.0];
        let mean_jk = jackknife_mean_variance(&data).unwrap();
        let id_jk = jackknife(&data, |d| d.iter().sum::<f64>() / d.len() as f64).unwrap();
        assert!((mean_jk.estimate - id_jk.estimate).abs() < 1e-9);
    }

    #[test]
    fn test_jackknife_result_std_error_is_sqrt_variance() {
        let data = [1.0, 3.0, 5.0];
        let r = jackknife_mean_variance(&data).unwrap();
        assert!((r.std_error - r.variance.sqrt()).abs() < JK_TOL);
    }

    #[test]
    fn test_jackknife_generalized_median_statistic() {
        fn median(d: &[f64]) -> f64 {
            let mut sorted = d.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let n = sorted.len();
            if n.is_multiple_of(2) {
                f64::midpoint(sorted[n / 2 - 1], sorted[n / 2])
            } else {
                sorted[n / 2]
            }
        }
        let data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = jackknife(&data, median).unwrap();
        assert!(result.variance >= 0.0);
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn test_jackknife_generalized_two_elements() {
        let data = [10.0, 20.0];
        let result = jackknife(&data, |d| d.iter().sum::<f64>() / d.len() as f64).unwrap();
        assert!((result.estimate - 15.0).abs() < 1e-8);
    }

    #[test]
    fn test_jackknife_mean_variance_all_identical_large() {
        let data = vec![7.0; 100];
        let r = jackknife_mean_variance(&data).unwrap();
        assert!((r.estimate - 7.0).abs() < JK_TOL);
        assert!(r.variance < 1e-20, "identical values → zero variance");
    }

    #[test]
    fn test_jackknife_mean_variance_two_values() {
        let data = [0.0, 100.0];
        let r = jackknife_mean_variance(&data).unwrap();
        assert!((r.estimate - 50.0).abs() < JK_TOL);
        assert!(r.variance > 0.0);
    }

    #[test]
    fn test_jackknife_result_debug_clone() {
        let r = JackknifeResult {
            estimate: 5.0,
            variance: 0.5,
            std_error: 0.5_f64.sqrt(),
        };
        let r2 = r;
        assert_eq!(r2.estimate, 5.0);
        let _ = format!("{r2:?}");
    }

    #[test]
    fn test_jackknife_generalized_sum_statistic() {
        let data = [1.0, 2.0, 3.0];
        let result = jackknife(&data, |d| d.iter().sum::<f64>()).unwrap();
        assert!(result.variance >= 0.0);
    }

    #[test]
    fn test_jackknife_generalized_max_statistic() {
        let data = [1.0, 5.0, 3.0, 4.0, 2.0];
        let result = jackknife(&data, |d| {
            d.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        })
        .unwrap();
        assert!(result.std_error >= 0.0);
    }

    #[test]
    fn test_jackknife_mean_linear_data() {
        let data: Vec<f64> = (1..=50).map(|i| i as f64).collect();
        let r = jackknife_mean_variance(&data).unwrap();
        assert!((r.estimate - 25.5).abs() < 1e-10);
    }

    #[cfg(feature = "gpu")]
    #[tokio::test]
    async fn test_jackknife_gpu_large_dataset() {
        use crate::device::test_pool::test_prelude::*;
        let Some(device) = test_f64_device().await else {
            return;
        };
        let data: Vec<f64> = (0..200).map(|i| (i as f64) * 0.1).collect();
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
        assert!(
            (cpu_result.estimate - gpu_result.estimate).abs() < 1e-6,
            "CPU={} GPU={}",
            cpu_result.estimate,
            gpu_result.estimate
        );
    }
}
