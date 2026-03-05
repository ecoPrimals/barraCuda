// SPDX-License-Identifier: AGPL-3.0-or-later
//! Weighted Dot Product (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Computes weighted inner products: result = Σ_k w[k] · a[k] · b[k]
//!
//! **Use cases**:
//! - Galerkin methods: <φ_i|W|φ_j>
//! - FEM assembly: element matrices
//! - Nuclear physics: potential matrix elements
//! - Energy integrals: ∫ρ(r)V(r)dr via quadrature

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/reduce/weighted_dot_f64.wgsl");

/// Parameters for weighted dot product
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DotParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated f64 weighted dot product — pipeline-cached, buffer-pooled
pub struct WeightedDotF64 {
    device: Arc<WgpuDevice>,
}

impl WeightedDotF64 {
    /// Create a new WeightedDotF64 orchestrator
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute weighted dot product: Σ w[i] * a[i] * b[i]
    ///
    /// # Arguments
    /// * `weights` - Weight vector
    /// * `a` - First vector
    /// * `b` - Second vector
    ///
    /// # Returns
    /// The weighted dot product as a single f64
    pub fn weighted_dot(&self, weights: &[f64], a: &[f64], b: &[f64]) -> Result<f64> {
        let n = weights.len();
        if a.len() != n || b.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Vector lengths must match: weights={}, a={}, b={}",
                    n,
                    a.len(),
                    b.len()
                ),
            });
        }

        if n == 0 {
            return Ok(0.0);
        }

        self.weighted_dot_gpu(weights, a, b)
    }

    /// CPU reference implementation
    #[cfg(test)]
    fn weighted_dot_cpu(&self, weights: &[f64], a: &[f64], b: &[f64]) -> f64 {
        weights
            .iter()
            .zip(a.iter())
            .zip(b.iter())
            .map(|((w, a), b)| w * a * b)
            .sum()
    }

    /// Unweighted dot product: Σ a[i] * b[i]
    pub fn dot(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        let n = a.len();
        if b.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!("Vector lengths must match: a={}, b={}", n, b.len()),
            });
        }

        if n == 0 {
            return Ok(0.0);
        }

        let ones = vec![1.0f64; n];
        self.weighted_dot_gpu(&ones, a, b)
    }

    /// Squared L2 norm: Σ a[i]²
    pub fn norm_squared(&self, a: &[f64]) -> Result<f64> {
        self.dot(a, a)
    }

    fn weighted_dot_gpu(&self, weights: &[f64], a: &[f64], b: &[f64]) -> Result<f64> {
        let n = weights.len();
        let workgroup_size = 256;
        let n_workgroups = n.div_ceil(workgroup_size);
        let ctx = get_device_context(&self.device);
        let adapter_info = self.device.adapter_info();

        let weights_buf =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("WDot Weights"),
                    contents: bytemuck::cast_slice(weights),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let a_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WDot A"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WDot B"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buf = ctx.acquire_pooled_output_f64(n_workgroups);

        let params = DotParams {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = self.device.create_uniform_buffer("WDot Params", &params);

        let layout_sig = BindGroupLayoutSignature::three_input_reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&weights_buf, &a_buf, &b_buf, &output_buf, &params_buf],
            Some("WDot BG"),
        );

        let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
            self.device.device(),
            adapter_info,
            SHADER,
            layout_sig,
            "weighted_dot_parallel",
            Some("WDot Pipeline"),
        );

        let wg = n_workgroups as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WDot Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        })?;

        let partial_sums: Vec<f64> = self.device.read_buffer_f64(&output_buf, n_workgroups)?;
        let result: f64 = partial_sums.iter().sum();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_weighted_dot_small() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = WeightedDotF64::new(device).unwrap();

        let w = vec![1.0, 2.0, 3.0];
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0, 1.0];

        let result = op.weighted_dot(&w, &a, &b).unwrap();
        assert!((result - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = WeightedDotF64::new(device).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];

        let result = op.dot(&a, &b).unwrap();
        assert!((result - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_norm_squared() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = WeightedDotF64::new(device).unwrap();

        let a = vec![3.0, 4.0];

        let result = op.norm_squared(&a).unwrap();
        assert!((result - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_dot_large() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = WeightedDotF64::new(device).unwrap();

        let n = 10_000;
        let w: Vec<f64> = (0..n).map(|i| 1.0 / (i as f64 + 1.0)).collect();
        let a: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

        let gpu_result = op.weighted_dot(&w, &a, &b).unwrap();
        let cpu_result = op.weighted_dot_cpu(&w, &a, &b);

        assert!(
            (gpu_result - cpu_result).abs() < 1e-8,
            "GPU: {}, CPU: {}",
            gpu_result,
            cpu_result
        );
    }
}
