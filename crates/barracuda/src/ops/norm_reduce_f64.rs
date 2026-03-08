// SPDX-License-Identifier: AGPL-3.0-or-later
//! Norm Reduction (f64) — GPU-Accelerated via WGSL
//!
//! Computes vector norms over f64 arrays:
//! - L1 norm: sum(|x|)
//! - L2 norm: sqrt(sum(x^2))
//! - Linf norm: max(|x|)
//! - Frobenius norm: `sqrt(sum(|a_ij|^2))` for matrices
//! - Generic p-norm: (sum(|x|^p))^(1/p)
//!
//! **Use cases**:
//! - Convergence checking (||residual||)
//! - Error metrics
//! - Regularization terms
//! - Scientific computing
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision via SPIR-V/Vulkan
//! - Safe Rust wrapper (no unsafe code)

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for norm shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct NormParams {
    size: u32,
    norm_type: u32, // 1=L1, 2=L2, 0=Linf
    p_lo: u32,      // f64 p as two u32s (for p-norm)
    p_hi: u32,
}

/// Native f64 norm-reduce shader (workgroup shared memory uses f64).
const SHADER_NATIVE: &str = include_str!("../shaders/reduce/norm_reduce_f64.wgsl");
/// DF64 norm-reduce shader (workgroup shared memory uses f32 pairs).
const SHADER_DF64: &str = include_str!("../shaders/reduce/norm_reduce_df64.wgsl");
/// DF64 core arithmetic library.
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select the reduce shader based on the device's FP64 strategy.
///
/// Native/Concurrent/Sovereign: native f64 workgroup memory is reliable.
/// Hybrid: f64 shared memory returns zeros on some devices; use DF64.
fn shader_for_device(device: &WgpuDevice) -> &'static str {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => SHADER_NATIVE,
        Fp64Strategy::Hybrid => {
            static DF64_COMBINED: std::sync::LazyLock<String> =
                std::sync::LazyLock::new(|| format!("enable f64;\n{DF64_CORE}\n{SHADER_DF64}"));
            &DF64_COMBINED
        }
    }
}

/// GPU-accelerated f64 norm operations
pub struct NormReduceF64;

impl NormReduceF64 {
    /// Compute L1 norm: sum(|x|)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or readback fails (e.g., device lost).
    pub fn l1(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "norm_l1_f64", None)
    }

    /// Compute L2 norm: sqrt(sum(x^2))
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or readback fails (e.g., device lost).
    pub fn l2(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let sum_sq = Self::reduce_op(device, data, "norm_l2_f64", None)?;
        Ok(sum_sq.sqrt())
    }

    /// Compute L2 norm squared: sum(x^2) (without sqrt)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or readback fails (e.g., device lost).
    pub fn l2_squared(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "norm_l2_f64", None)
    }

    /// Compute Linf norm: max(|x|)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or readback fails (e.g., device lost).
    pub fn linf(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "norm_linf_f64", None)
    }

    /// Compute Frobenius norm (same as L2, but semantically for matrices)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or readback fails (e.g., device lost).
    pub fn frobenius(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let sum_sq = Self::reduce_op(device, data, "norm_frobenius_f64", None)?;
        Ok(sum_sq.sqrt())
    }

    /// Compute generic p-norm: (sum(|x|^p))^(1/p)
    /// Dispatches `norm_p_f64` WGSL shader. `compile_shader_f64()` auto-injects
    /// software `pow_f64` on drivers lacking native f64 pow (NVK, RADV, Ada).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or readback fails (e.g., device lost).
    pub fn p_norm(device: Arc<WgpuDevice>, data: &[f64], p: f64) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        if p == 1.0 {
            return Self::l1(device, data);
        }
        if p == 2.0 {
            return Self::l2(device, data);
        }
        if p.is_infinite() && p > 0.0 {
            return Self::linf(device, data);
        }

        let sum_p = Self::reduce_op(device, data, "norm_p_f64", Some(p))?;
        Ok(sum_p.powf(1.0 / p))
    }

    fn reduce_op(
        device: Arc<WgpuDevice>,
        data: &[f64],
        entry_point: &str,
        p: Option<f64>,
    ) -> Result<f64> {
        if data.is_empty() {
            return Ok(0.0);
        }
        if data.len() == 1 {
            return Ok(data[0].abs());
        }

        let n = data.len();
        let wg_size = 256;
        let n_workgroups = n.div_ceil(wg_size);

        // Create input buffer
        let input_bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NormReduce input"),
                contents: &input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let partial_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NormReduce partials"),
            size: (n_workgroups * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Pack p as two u32s (low and high bits of f64)
        let p_val = p.unwrap_or(2.0);
        let p_bits = p_val.to_bits();
        let params = NormParams {
            size: n as u32,
            norm_type: 0,
            p_lo: p_bits as u32,
            p_hi: (p_bits >> 32) as u32,
        };
        let params_buffer = device.create_uniform_buffer("NormReduce params", &params);

        ComputeDispatch::new(&device, "norm_reduce_pass1")
            .shader(shader_for_device(&device), entry_point)
            .f64()
            .storage_read(0, &input_buffer)
            .storage_rw(1, &partial_buffer)
            .uniform(2, &params_buffer)
            .dispatch(n_workgroups as u32, 1, 1)
            .submit()?;

        if n_workgroups <= 1 {
            return Self::read_f64_scalar(&device, &partial_buffer);
        }

        // Second pass: use sum reduction for L1/L2/p-norm, max for Linf
        let second_entry = if entry_point == "norm_linf_f64" {
            "norm_linf_f64"
        } else {
            "norm_l1_f64" // L1 is just sum for second pass
        };

        let final_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NormReduce final"),
            size: 8 * n_workgroups.div_ceil(wg_size) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params2 = NormParams {
            size: n_workgroups as u32,
            norm_type: 0,
            p_lo: 0,
            p_hi: 0,
        };
        let params2_buffer = device.create_uniform_buffer("NormReduce params 2", &params2);

        let n_workgroups2 = n_workgroups.div_ceil(wg_size);
        ComputeDispatch::new(&device, "norm_reduce_pass2")
            .shader(shader_for_device(&device), second_entry)
            .f64()
            .storage_read(0, &partial_buffer)
            .storage_rw(1, &final_buffer)
            .uniform(2, &params2_buffer)
            .dispatch(n_workgroups2 as u32, 1, 1)
            .submit()?;

        if n_workgroups2 > 1 {
            let partials = device.read_f64_buffer(&final_buffer, n_workgroups2)?;
            if entry_point == "norm_linf_f64" {
                return Ok(partials.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
            }
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
    async fn test_l1_norm() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, -2.0, 3.0, -4.0, 5.0];
        let norm = NormReduceF64::l1(device, &data).unwrap();
        assert!(
            (norm - 15.0).abs() < 1e-6,
            "L1 norm should be 15, got {norm}"
        );
    }

    #[tokio::test]
    async fn test_l2_norm() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![3.0, 4.0];
        let norm = NormReduceF64::l2(device, &data).unwrap();
        assert!(
            (norm - 5.0).abs() < 1e-6,
            "L2 norm of [3,4] should be 5, got {norm}"
        );
    }

    #[tokio::test]
    async fn test_linf_norm() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, -7.0, 3.0, -4.0, 5.0];
        let norm = NormReduceF64::linf(device, &data).unwrap();
        assert!(
            (norm - 7.0).abs() < 1e-6,
            "Linf norm should be 7, got {norm}"
        );
    }

    #[tokio::test]
    async fn test_frobenius_norm() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // 2x2 identity matrix has Frobenius norm sqrt(2)
        let data: Vec<f64> = vec![1.0, 0.0, 0.0, 1.0];
        let norm = NormReduceF64::frobenius(device, &data).unwrap();
        let expected = 2.0_f64.sqrt();
        assert!(
            (norm - expected).abs() < 1e-6,
            "Frobenius norm should be sqrt(2), got {norm}"
        );
    }

    #[tokio::test]
    async fn test_p_norm() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        // p=3 norm: (1^3 + 2^3 + 3^3 + 4^3)^(1/3) = (1+8+27+64)^(1/3) = 100^(1/3)
        let norm = NormReduceF64::p_norm(device, &data, 3.0).unwrap();
        let expected = 100.0_f64.powf(1.0 / 3.0);
        assert!(
            (norm - expected).abs() < 1e-4,
            "p=3 norm should be {expected}, got {norm}"
        );
    }
}
