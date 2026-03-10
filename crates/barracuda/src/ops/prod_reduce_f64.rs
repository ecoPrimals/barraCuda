// SPDX-License-Identifier: AGPL-3.0-only
//! Product Reduction (f64) — GPU-Accelerated via WGSL
//!
//! Computes product over all elements of an f64 buffer.
//! Two-pass reduction: first pass produces partial products per workgroup,
//! second pass reduces partial products to a single scalar.
//!
//! **Use cases**:
//! - Determinant computation
//! - Probability product chains
//! - Partition function terms
//! - Any global f64 product
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

/// Native f64 product-reduce shader (workgroup shared memory uses f64).
const SHADER_NATIVE: &str = include_str!("../shaders/reduce/prod_reduce_f64.wgsl");
/// DF64 product-reduce shader (workgroup shared memory uses f32 pairs).
/// Compiled via `compile_shader_df64` which prepends the DF64 core preamble.
const SHADER_DF64: &str = include_str!("../shaders/reduce/prod_reduce_df64.wgsl");

/// Select the reduce shader and compilation mode based on the device's FP64 strategy.
fn shader_config_for_device(device: &WgpuDevice) -> (&'static str, bool) {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
            (SHADER_NATIVE, false)
        }
        Fp64Strategy::Hybrid => (SHADER_DF64, true),
    }
}

/// GPU-accelerated f64 product reduction operations
pub struct ProdReduceF64;

impl ProdReduceF64 {
    /// Compute the product of all elements in a f64 buffer on GPU
    ///
    /// # Arguments
    /// * `device` - `WgpuDevice`
    /// * `data` - Input f64 slice
    ///
    /// # Returns
    /// The product as a single f64
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn prod(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        Self::reduce_op(device, data, "prod_reduce_f64")
    }

    /// Compute the product using log domain (numerically stable for very large products)
    ///
    /// Computes exp(sum(log(x))) which is equivalent to product(x) but more stable
    /// for long sequences where the direct product would overflow.
    ///
    /// # Note
    /// Only works for positive values. Returns NaN for negative or zero inputs.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn log_prod(device: Arc<WgpuDevice>, data: &[f64]) -> Result<f64> {
        let log_sum = Self::reduce_op(device, data, "log_prod_reduce_f64")?;
        Ok(log_sum.exp())
    }

    fn reduce_op(device: Arc<WgpuDevice>, data: &[f64], entry_point: &str) -> Result<f64> {
        if data.is_empty() {
            // Identity for product is 1.0
            return Ok(1.0);
        }
        if data.len() == 1 {
            return Ok(data[0]);
        }

        // Two-pass reduction
        let n = data.len();
        let wg_size = WORKGROUP_SIZE_1D as usize;
        let n_workgroups = n.div_ceil(wg_size);

        // Pass 1: data -> partial products
        let input_bytes: &[u8] = bytemuck::cast_slice(data);
        let input_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ProdReduce input"),
                contents: input_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let partial_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProdReduce partials"),
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
        let params_buffer = device.create_uniform_buffer("ProdReduce params", &params);

        let (shader_src, use_df64) = shader_config_for_device(&device);

        let mut pass1 = ComputeDispatch::new(&device, "prod_reduce_pass1")
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
            label: Some("ProdReduce final"),
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
        let params2_buffer = device.create_uniform_buffer("ProdReduce params 2", &params2);

        let n_workgroups2 = n_workgroups.div_ceil(wg_size);
        let mut pass2 = ComputeDispatch::new(&device, "prod_reduce_pass2")
            .shader(shader_src, entry_point)
            .storage_read(0, &partial_buffer)
            .storage_rw(1, &final_buffer)
            .uniform(2, &params2_buffer)
            .dispatch(n_workgroups2 as u32, 1, 1);
        pass2 = if use_df64 { pass2.df64() } else { pass2.f64() };
        pass2.submit()?;

        if n_workgroups2 > 1 {
            // Third pass (extremely rare): read back partials and multiply on CPU
            let partials = device.read_f64_buffer(&final_buffer, n_workgroups2)?;
            return Ok(partials.iter().product());
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
    async fn test_prod_small() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let prod = ProdReduceF64::prod(device, &data).unwrap();
        assert!(
            (prod - 120.0).abs() < 1e-6,
            "Product of 1..5 should be 120, got {prod}"
        );
    }

    #[tokio::test]
    async fn test_prod_identity() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // Empty product should be 1 (identity)
        let prod = ProdReduceF64::prod(device, &[]).unwrap();
        assert!(
            (prod - 1.0).abs() < 1e-10,
            "Empty product should be 1, got {prod}"
        );
    }

    #[tokio::test]
    async fn test_prod_with_zeros() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let data: Vec<f64> = vec![1.0, 2.0, 0.0, 4.0, 5.0];
        let prod = ProdReduceF64::prod(device, &data).unwrap();
        assert!(
            prod.abs() < 1e-10,
            "Product with zero should be 0, got {prod}"
        );
    }

    #[tokio::test]
    async fn test_prod_large() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // Use small values to avoid overflow
        let data: Vec<f64> = vec![1.1; 100];
        let prod = ProdReduceF64::prod(device.clone(), &data).unwrap();
        let expected = 1.1_f64.powi(100);
        let rel_error = (prod - expected).abs() / expected;
        assert!(
            rel_error < 1e-6,
            "Large product error too high: {rel_error} (expected {expected}, got {prod})"
        );
    }
}
