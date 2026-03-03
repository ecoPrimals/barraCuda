//! Max Absolute Difference Reduction (f64) — GPU-Accelerated Convergence Check
//!
//! Computes `max|a[i] - b[i]|` over all elements of two f64 buffers.
//! Essential for iterative solver convergence checking.
//!
//! **Use cases**:
//! - SCF convergence: `max|E_new - E_old| < tolerance`
//! - Iterative solver termination
//! - Energy difference monitoring across batched systems
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision via SPIR-V/Vulkan
//! - Safe Rust wrapper (no unsafe code)
//!
//! # Example
//!
//! ```rust,ignore
//! let device = WgpuDevice::new().await?;
//! let diff = MaxAbsDiffF64::compute(&device, &e_new, &e_old)?;
//! if diff < 1e-10 {
//!     println!("Converged!");
//! }
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Parameters for max_abs_diff shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DiffParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

/// GPU-accelerated max absolute difference computation
///
/// Two-pass reduction: first pass computes partial maxes per workgroup,
/// second pass reduces partial maxes to a single scalar.
pub struct MaxAbsDiffF64;

impl MaxAbsDiffF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/reduce/max_abs_diff_f64.wgsl")
    }

    /// Compute max|a[i] - b[i]| over all elements
    ///
    /// # Arguments
    /// * `device` - WgpuDevice
    /// * `a` - First f64 slice
    /// * `b` - Second f64 slice (must have same length as `a`)
    ///
    /// # Returns
    /// The maximum absolute difference as a single f64
    ///
    /// # Errors
    /// Returns error if arrays have different lengths
    pub fn compute(device: Arc<WgpuDevice>, a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "MaxAbsDiffF64: array lengths must match ({} vs {})",
                    a.len(),
                    b.len()
                ),
            });
        }

        if a.is_empty() {
            return Ok(0.0);
        }
        if a.len() == 1 {
            return Ok((a[0] - b[0]).abs());
        }

        // Two-pass reduction
        let n = a.len();
        let wg_size = 256;
        let n_workgroups = n.div_ceil(wg_size);

        // Create input buffers
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        let a_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MaxAbsDiff input_a"),
                contents: &a_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        let b_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MaxAbsDiff input_b"),
                contents: &b_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        // Partial results buffer (one max per workgroup)
        let partial_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxAbsDiff partials"),
            size: (n_workgroups * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = DiffParams {
            size: n as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params_buffer = device.create_uniform_buffer("MaxAbsDiff params", &params);

        ComputeDispatch::new(&device, "max_abs_diff_pass1")
            .shader(Self::wgsl_shader(), "max_abs_diff_f64")
            .f64()
            .storage_read(0, &a_buffer)
            .storage_read(1, &b_buffer)
            .storage_rw(2, &partial_buffer)
            .uniform(3, &params_buffer)
            .dispatch(n_workgroups as u32, 1, 1)
            .submit();

        // If single workgroup, result is ready
        if n_workgroups <= 1 {
            return Self::read_f64_scalar(&device, &partial_buffer);
        }

        let n_workgroups2 = n_workgroups.div_ceil(wg_size);
        let final_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxAbsDiff final"),
            size: (n_workgroups2 * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params2 = DiffParams {
            size: n_workgroups as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };
        let params2_buffer = device.create_uniform_buffer("MaxAbsDiff params 2", &params2);

        // For pass 2, partial_buffer becomes input_a, we need a dummy input_b
        let dummy_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MaxAbsDiff dummy"),
            size: (n_workgroups * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(&device, "max_abs_diff_pass2")
            .shader(Self::wgsl_shader(), "max_reduce_pass2")
            .f64()
            .storage_read(0, &partial_buffer)
            .storage_read(1, &dummy_buffer)
            .storage_rw(2, &final_buffer)
            .uniform(3, &params2_buffer)
            .dispatch(n_workgroups2 as u32, 1, 1)
            .submit();

        // For very large inputs, may need CPU fallback
        if n_workgroups2 > 1 {
            let partials = device.read_f64_buffer(&final_buffer, n_workgroups2)?;
            return Ok(partials.iter().cloned().fold(0.0_f64, f64::max));
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
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_small() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let a = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.1_f64, 1.9, 3.5, 3.8, 5.0];
        // Differences: 0.1, 0.1, 0.5, 0.2, 0.0
        // Max = 0.5

        let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();
        assert!(
            (diff - 0.5).abs() < 1e-10,
            "Max abs diff should be 0.5, got {}",
            diff
        );
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_large() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // 2048 elements (multiple workgroups)
        let a: Vec<f64> = (0..2048).map(|i| i as f64).collect();
        let mut b: Vec<f64> = (0..2048).map(|i| i as f64).collect();
        // Make one element differ by 100
        b[1000] = a[1000] + 100.0;

        let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();
        assert!(
            (diff - 100.0).abs() < 1e-6,
            "Max abs diff should be 100, got {}",
            diff
        );
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_identical() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let a: Vec<f64> = (1..=100).map(|i| i as f64 * std::f64::consts::PI).collect();
        let b = a.clone();

        let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();
        assert!(
            diff < 1e-14,
            "Max abs diff of identical arrays should be ~0, got {}",
            diff
        );
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_empty() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];

        let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();
        assert!((diff - 0.0).abs() < 1e-14, "Empty should return 0");
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_single_element() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let a = vec![5.0_f64];
        let b = vec![3.0_f64];

        let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();
        assert!(
            (diff - 2.0).abs() < 1e-14,
            "Single element diff should be 2"
        );
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_length_mismatch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let a = vec![1.0_f64, 2.0];
        let b = vec![1.0_f64];

        let result = MaxAbsDiffF64::compute(device, &a, &b);
        assert!(result.is_err(), "Should fail on length mismatch");
    }

    #[tokio::test]
    #[ignore = "requires GPU hardware"]
    async fn test_max_abs_diff_negative_values() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let a = vec![-5.0_f64, -10.0, 3.0];
        let b = vec![5.0_f64, -10.0, -3.0];
        // Differences: 10, 0, 6 -> max = 10

        let diff = MaxAbsDiffF64::compute(device, &a, &b).unwrap();
        assert!(
            (diff - 10.0).abs() < 1e-10,
            "Max abs diff should be 10, got {}",
            diff
        );
    }
}
