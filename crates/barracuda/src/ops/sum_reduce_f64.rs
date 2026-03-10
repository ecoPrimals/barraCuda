// SPDX-License-Identifier: AGPL-3.0-only
//! Sum Reduction (f64) — GPU-Accelerated via WGSL
//!
//! Computes sum, max, or min over all elements of an f64 buffer.
//! Two-pass reduction: first pass produces partial sums per workgroup,
//! second pass reduces partial sums to a single scalar.
//!
//! **Use cases**:
//! - Energy functional integration (trapezoid rule: sum of integrand * dr)
//! - RMS error computation: sqrt(sum(errors^2) / N)
//! - Convergence checking: `max(|delta_E`|)
//! - Any global f64 reduction
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision via SPIR-V/Vulkan
//! - Safe Rust wrapper (no unsafe code)

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for reduce shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// Native f64 sum-reduce shader (workgroup shared memory uses f64).
const SHADER_NATIVE: &str = include_str!("../shaders/reduce/sum_reduce_f64.wgsl");
/// DF64 sum-reduce shader (workgroup shared memory uses f32 pairs).
/// Compiled via `compile_shader_df64` which prepends the DF64 core preamble.
const SHADER_DF64: &str = include_str!("../shaders/reduce/sum_reduce_df64.wgsl");

/// Select the reduce shader and compilation mode based on the device's FP64 strategy.
///
/// Native/Concurrent/Sovereign: native f64 workgroup memory is reliable → `.f64()`.
/// Hybrid: f64 shared memory returns zeros on some devices → DF64 source + `.df64()`.
fn shader_config_for_device(device: &WgpuDevice) -> (&'static str, bool) {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
            (SHADER_NATIVE, false)
        }
        Fp64Strategy::Hybrid => (SHADER_DF64, true),
    }
}

/// GPU-accelerated f64 reduction operations
pub struct SumReduceF64;

impl SumReduceF64 {
    /// Compute the sum of all elements in a f64 buffer on GPU
    /// # Arguments
    /// * `device` - `WgpuDevice`
    /// * `data` - Input f64 slice
    /// # Returns
    /// The sum as a single f64
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn sum(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "sum_reduce_f64")
    }

    /// Compute the max of all elements in a f64 buffer on GPU
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn max(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "max_reduce_f64")
    }

    /// Compute the min of all elements in a f64 buffer on GPU
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn min(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "min_reduce_f64")
    }

    /// Compute the mean of all elements
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn mean(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let sum = Self::sum(device, data)?;
        Ok(sum / data.len() as f64)
    }

    fn reduce_op(device: Arc<WgpuDevice>, data: &[f64], entry_point: &str) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        if data.len() == 1 {
            return Ok(data[0]);
        }

        let (shader_src, use_df64) = shader_config_for_device(&device);

        let n = data.len();
        let wg_size = WORKGROUP_SIZE_1D as usize;
        let n_workgroups = n.div_ceil(wg_size);

        let input_bytes: &[u8] = bytemuck::cast_slice(data);
        let input_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Reduce input"),
                contents: input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let partial_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduce partials"),
            size: (n_workgroups * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = ReduceParams {
            size: n as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer = device.create_uniform_buffer("Reduce params", &params);

        let mut pass1 = ComputeDispatch::new(&device, "sum_reduce_pass1")
            .shader(shader_src, entry_point)
            .storage_read(0, &input_buffer)
            .storage_rw(1, &partial_buffer)
            .uniform(2, &params_buffer)
            .dispatch(n_workgroups as u32, 1, 1);
        pass1 = if use_df64 { pass1.df64() } else { pass1.f64() };
        pass1.submit()?;

        if n_workgroups <= 1 {
            return Self::read_f64_scalar(&device, &partial_buffer);
        }

        let final_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reduce final"),
            size: 8 * n_workgroups.div_ceil(wg_size) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params2 = ReduceParams {
            size: n_workgroups as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params2_buffer = device.create_uniform_buffer("Reduce params 2", &params2);

        let n_workgroups2 = n_workgroups.div_ceil(wg_size);
        let mut pass2 = ComputeDispatch::new(&device, "sum_reduce_pass2")
            .shader(shader_src, entry_point)
            .storage_read(0, &partial_buffer)
            .storage_rw(1, &final_buffer)
            .uniform(2, &params2_buffer)
            .dispatch(n_workgroups2 as u32, 1, 1);
        pass2 = if use_df64 { pass2.df64() } else { pass2.f64() };
        pass2.submit()?;

        if n_workgroups2 > 1 {
            let partials = device.read_f64_buffer(&final_buffer, n_workgroups2)?;
            return Ok(partials.iter().sum());
        }

        Self::read_f64_scalar(&device, &final_buffer)
    }

    fn read_f64_scalar(device: &WgpuDevice, buffer: &wgpu::Buffer) -> Result<f64> {
        let values = device.read_f64_buffer(buffer, 1)?;
        Ok(values[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sum_small() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let sum = SumReduceF64::sum(device, &data).unwrap();
        assert!(
            (sum - 5050.0).abs() < 1e-6,
            "Sum of 1..100 should be 5050, got {sum}"
        );
    }

    #[tokio::test]
    async fn test_sum_large() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // 2048 elements (multiple workgroups)
        let data: Vec<f64> = (1..=2048).map(|i| i as f64).collect();
        let expected = 2048.0 * 2049.0 / 2.0;
        let sum = SumReduceF64::sum(device, &data).unwrap();
        assert!(
            (sum - expected).abs() < 1e-3,
            "Sum of 1..2048 should be {expected}, got {sum}"
        );
    }

    #[tokio::test]
    async fn test_max() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data = vec![3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let max = SumReduceF64::max(device, &data).unwrap();
        assert!((max - 9.0).abs() < 1e-10, "Max should be 9, got {max}");
    }
}
