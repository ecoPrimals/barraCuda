// SPDX-License-Identifier: AGPL-3.0-or-later
//! Variance/Std Reduction (f64) — GPU-Accelerated via WGSL
//!
//! Computes variance and standard deviation of f64 arrays using
//! Welford's numerically stable online algorithm.
//!
//! **Use cases**:
//! - Statistical analysis
//! - Error estimation (standard error)
//! - Convergence metrics
//! - Scientific measurements
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision via SPIR-V/Vulkan
//! - Numerically stable (Welford's algorithm)
//! - Safe Rust wrapper (no unsafe code)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Parameters for reduce shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Welford state for parallel merging
#[derive(Clone, Copy, Debug)]
struct WelfordState {
    count: f64,
    mean: f64,
    m2: f64,
}

impl WelfordState {
    fn merge(self, other: WelfordState) -> WelfordState {
        let count = self.count + other.count;
        if count == 0.0 {
            return WelfordState {
                count: 0.0,
                mean: 0.0,
                m2: 0.0,
            };
        }
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * other.count / count;
        let m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / count;
        WelfordState { count, mean, m2 }
    }
}

/// GPU-accelerated f64 variance/std operations
pub struct VarianceReduceF64;

impl VarianceReduceF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/reduce/variance_reduce_f64.wgsl")
    }

    /// Compute sample variance: Var(X) = sum((x - mean)^2) / (n - 1)
    ///
    /// Uses Bessel's correction (n-1 denominator) for unbiased estimation.
    pub fn variance(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let state = Self::compute_welford_state(device, data)?;
        if state.count < 2.0 {
            return Ok(0.0);
        }
        Ok(state.m2 / (state.count - 1.0))
    }

    /// Compute population variance: Var(X) = sum((x - mean)^2) / n
    pub fn population_variance(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let state = Self::compute_welford_state(device, data)?;
        if state.count < 1.0 {
            return Ok(0.0);
        }
        Ok(state.m2 / state.count)
    }

    /// Compute sample standard deviation: sqrt(variance)
    pub fn std(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let var = Self::variance(device, data)?;
        Ok(var.sqrt())
    }

    /// Compute population standard deviation
    pub fn population_std(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let var = Self::population_variance(device, data)?;
        Ok(var.sqrt())
    }

    /// Compute mean using Welford state (bonus: numerically stable)
    pub fn mean(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let state = Self::compute_welford_state(device, data)?;
        Ok(state.mean)
    }

    /// Compute both mean and variance in one pass
    pub fn mean_and_variance(device: Arc<WgpuDevice>, data: &[f64]) -> Result<(f64, f64)> {
        let state = Self::compute_welford_state(device, data)?;
        let var = if state.count < 2.0 {
            0.0
        } else {
            state.m2 / (state.count - 1.0)
        };
        Ok((state.mean, var))
    }

    /// Compute all statistics: (count, mean, variance, std)
    pub fn statistics(device: Arc<WgpuDevice>, data: &[f64]) -> Result<(usize, f64, f64, f64)> {
        let state = Self::compute_welford_state(device, data)?;
        let var = if state.count < 2.0 {
            0.0
        } else {
            state.m2 / (state.count - 1.0)
        };
        Ok((state.count as usize, state.mean, var, var.sqrt()))
    }

    fn compute_welford_state(device: Arc<WgpuDevice>, data: &[f64]) -> Result<WelfordState> {
        if data.is_empty() {
            return Ok(WelfordState {
                count: 0.0,
                mean: 0.0,
                m2: 0.0,
            });
        }
        if data.len() == 1 {
            return Ok(WelfordState {
                count: 1.0,
                mean: data[0],
                m2: 0.0,
            });
        }

        let n = data.len();
        let wg_size = 256;
        let n_workgroups = n.div_ceil(wg_size);

        // Create input buffer
        let input_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VarianceReduce input"),
                contents: &input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Output: 3 f64s per workgroup (count, mean, M2)
        let partial_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VarianceReduce partials"),
            size: (n_workgroups * 3 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = ReduceParams {
            size: n as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer = device.create_uniform_buffer("VarianceReduce params", &params);

        ComputeDispatch::new(&device, "variance_reduce_f64")
            .shader(Self::wgsl_shader(), "variance_reduce_f64")
            .f64()
            .storage_read(0, &input_buffer)
            .storage_rw(1, &partial_buffer)
            .uniform(2, &params_buffer)
            .dispatch(n_workgroups as u32, 1, 1)
            .submit();

        // Read back partial states and merge on CPU
        let partials = device.read_f64_buffer(&partial_buffer, n_workgroups * 3)?;

        // Merge all partial Welford states
        let mut final_state = WelfordState {
            count: 0.0,
            mean: 0.0,
            m2: 0.0,
        };
        for chunk in partials.chunks_exact(3) {
            let state = WelfordState {
                count: chunk[0],
                mean: chunk[1],
                m2: chunk[2],
            };
            final_state = final_state.merge(state);
        }

        Ok(final_state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_variance_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let var = VarianceReduceF64::variance(device, &data).unwrap();
        // Expected sample variance: 4.571...
        let expected = 4.571428571428571;
        assert!(
            (var - expected).abs() < 1e-6,
            "Variance should be ~{}, got {}",
            expected,
            var
        );
    }

    #[tokio::test]
    async fn test_std_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = VarianceReduceF64::std(device, &data).unwrap();
        let expected = 4.571428571428571_f64.sqrt();
        assert!(
            (std - expected).abs() < 1e-6,
            "Std should be ~{}, got {}",
            expected,
            std
        );
    }

    #[tokio::test]
    async fn test_mean() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = VarianceReduceF64::mean(device, &data).unwrap();
        assert!((mean - 3.0).abs() < 1e-6, "Mean should be 3, got {}", mean);
    }

    #[tokio::test]
    async fn test_population_vs_sample_variance() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample_var = VarianceReduceF64::variance(device.clone(), &data).unwrap();
        let pop_var = VarianceReduceF64::population_variance(device, &data).unwrap();

        // Population variance uses n, sample uses n-1
        // So sample_var * (n-1) / n = pop_var
        let n = data.len() as f64;
        let expected_ratio = (n - 1.0) / n;
        let actual_ratio = pop_var / sample_var;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-6,
            "Variance ratio should be (n-1)/n, got {}",
            actual_ratio
        );
    }

    #[tokio::test]
    async fn test_statistics() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let (count, mean, var, std) = VarianceReduceF64::statistics(device, &data).unwrap();

        assert_eq!(count, 100);
        assert!((mean - 50.5).abs() < 1e-6);
        // Sample variance of 1..100: 841.666...
        assert!((var - 841.6666666666666).abs() < 1e-3);
        assert!((std - var.sqrt()).abs() < 1e-6);
    }
}
