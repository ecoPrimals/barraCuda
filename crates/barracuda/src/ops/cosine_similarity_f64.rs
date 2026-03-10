// SPDX-License-Identifier: AGPL-3.0-only
//! Cosine Similarity (f64) — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Computes cosine similarity: sim(a,b) = (a·b) / (||a|| * ||b||)
//!
//! **Use cases**:
//! - MS2 spectral matching in analytical chemistry
//! - High-precision similarity search
//! - Biological sequence comparison
//!
//! **Note**: For large batches, `GemmF64 + FusedMapReduceF64` is often faster.
//! This dedicated shader is optimal for single-pair or small-batch queries
//! where GEMM overhead dominates.

use crate::device::WgpuDevice;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::pipeline_cache::{BindGroupLayoutSignature, create_f64_data_pipeline};
use crate::device::tensor_context::get_device_context;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/math/cosine_similarity_f64.wgsl");
const DF64_CORE: &str = include_str!("../shaders/math/df64_core.wgsl");

/// Select shader based on FP64 strategy: native f64 or DF64 auto-rewrite.
///
/// On Hybrid devices the native f64 shader may silently produce zeros,
/// so we require the DF64 rewrite to succeed rather than falling back.
fn shader_for_device(device: &WgpuDevice) -> Result<&'static str> {
    let profile = GpuDriverProfile::from_device(device);
    match profile.fp64_strategy() {
        Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => Ok(SHADER),
        Fp64Strategy::Hybrid => {
            static DF64_RESULT: std::sync::LazyLock<std::result::Result<String, Arc<str>>> =
                std::sync::LazyLock::new(|| {
                    crate::shaders::sovereign::df64_rewrite::rewrite_f64_infix_full(SHADER)
                        .map(|src| format!("enable f64;\n{DF64_CORE}\n{src}"))
                        .map_err(|e| {
                            Arc::from(
                                format!("cosine_similarity DF64 rewrite failed: {e}").as_str(),
                            )
                        })
                });
            match DF64_RESULT.as_ref() {
                Ok(src) => Ok(src.as_str()),
                Err(msg) => Err(crate::error::BarracudaError::ShaderCompilation(Arc::clone(
                    msg,
                ))),
            }
        }
    }
}

/// Parameters for cosine similarity shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CosineParams {
    num_vectors_a: u32,
    num_vectors_b: u32,
    vector_dim: u32,
    _padding: u32,
}

/// GPU-accelerated f64 cosine similarity — pipeline-cached, buffer-pooled
pub struct CosineSimilarityF64 {
    device: Arc<WgpuDevice>,
}

impl CosineSimilarityF64 {
    /// Create a new `CosineSimilarityF64` orchestrator
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute cosine similarity between two single vectors
    /// # Arguments
    /// * `a` - First vector (f64)
    /// * `b` - Second vector (f64)
    /// # Returns
    /// Cosine similarity in [-1, 1]
    /// # Errors
    /// Returns [`Err`] if vector dimensions differ, vectors are empty, buffer allocation fails,
    /// GPU dispatch fails, buffer readback fails, or the device is lost.
    pub fn similarity(&self, a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!("Vector dimensions must match: a={}, b={}", a.len(), b.len()),
            });
        }

        let n = a.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "Empty vectors".to_string(),
            });
        }

        let matrix = self.dispatch_flat(a, b, 1, 1, n)?;
        Ok(matrix[0])
    }

    /// Compute all-pairs cosine similarity matrix
    /// # Arguments
    /// * `vectors_a` - First set of vectors (each of dimension `dim`)
    /// * `vectors_b` - Second set of vectors (each of dimension `dim`)
    /// * `dim` - Vector dimension
    /// # Returns
    /// Similarity matrix of shape [`len(vectors_a)`, `len(vectors_b)`]
    /// Row-major: result[i * `len(vectors_b)` + j] = `sim(vectors_a`[i], `vectors_b`[j])
    /// # Errors
    /// Returns [`Err`] if any vector has wrong dimension, buffer allocation fails, GPU dispatch fails,
    /// buffer readback fails, or the device is lost.
    pub fn all_pairs(
        &self,
        vectors_a: &[Vec<f64>],
        vectors_b: &[Vec<f64>],
        dim: usize,
    ) -> Result<Vec<f64>> {
        if vectors_a.is_empty() || vectors_b.is_empty() {
            return Ok(vec![]);
        }

        for (i, v) in vectors_a.iter().enumerate() {
            if v.len() != dim {
                return Err(BarracudaError::InvalidInput {
                    message: format!("vectors_a[{}] has dim {}, expected {}", i, v.len(), dim),
                });
            }
        }
        for (i, v) in vectors_b.iter().enumerate() {
            if v.len() != dim {
                return Err(BarracudaError::InvalidInput {
                    message: format!("vectors_b[{}] has dim {}, expected {}", i, v.len(), dim),
                });
            }
        }

        let a_flat: Vec<f64> = vectors_a.iter().flat_map(|v| v.iter().copied()).collect();
        let b_flat: Vec<f64> = vectors_b.iter().flat_map(|v| v.iter().copied()).collect();
        self.dispatch_flat(&a_flat, &b_flat, vectors_a.len(), vectors_b.len(), dim)
    }

    /// CPU reference implementation (single pair)
    #[allow(dead_code, reason = "CPU reference for GPU parity validation")]
    fn similarity_cpu(&self, a: &[f64], b: &[f64]) -> f64 {
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;

        for (ai, bi) in a.iter().zip(b.iter()) {
            dot += ai * bi;
            norm_a += ai * ai;
            norm_b += bi * bi;
        }

        let denom = (norm_a * norm_b).sqrt();
        if denom < crate::tolerances::eps::SAFE_DIV {
            return 0.0;
        }
        dot / denom
    }

    /// CPU reference implementation (all pairs)
    #[allow(dead_code, reason = "CPU reference for GPU parity validation")]
    fn all_pairs_cpu(&self, vectors_a: &[Vec<f64>], vectors_b: &[Vec<f64>]) -> Vec<f64> {
        let mut result = Vec::with_capacity(vectors_a.len() * vectors_b.len());
        for va in vectors_a {
            for vb in vectors_b {
                result.push(self.similarity_cpu(va, vb));
            }
        }
        result
    }

    fn dispatch_flat(
        &self,
        a_flat: &[f64],
        b_flat: &[f64],
        num_a: usize,
        num_b: usize,
        dim: usize,
    ) -> Result<Vec<f64>> {
        let ctx = get_device_context(&self.device);

        let a_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CosSim A"),
                contents: bytemuck::cast_slice(a_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let b_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CosSim B"),
                contents: bytemuck::cast_slice(b_flat),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = num_a * num_b;
        let output_buf = ctx.acquire_pooled_output_f64(output_size);

        let params = CosineParams {
            num_vectors_a: num_a as u32,
            num_vectors_b: num_b as u32,
            vector_dim: dim as u32,
            _padding: 0,
        };
        let params_buf = self.device.create_uniform_buffer("CosSim Params", &params);

        let layout_sig = BindGroupLayoutSignature::two_input_reduction();
        let bind_group = ctx.get_or_create_bind_group(
            layout_sig,
            &[&a_buf, &b_buf, &output_buf, &params_buf],
            Some("CosSim BG"),
        );

        let shader_src = shader_for_device(&self.device)?;
        let pipeline = create_f64_data_pipeline(
            &self.device,
            shader_src,
            layout_sig,
            "main",
            Some("CosSim Pipeline"),
        );

        let wg_x = num_a.div_ceil(16) as u32;
        let wg_y = num_b.div_ceil(16) as u32;
        ctx.record_operation(move |encoder| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CosSim Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&*bind_group), &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        })?;

        self.device.read_buffer_f64(&output_buf, output_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_identical_vectors() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = CosineSimilarityF64::new(device).unwrap();

        let a = vec![1.0, 2.0, 3.0, 4.0];
        let sim = op.similarity(&a, &a).unwrap();

        assert!(
            (sim - 1.0).abs() < 1e-9,
            "Expected 1.0 for identical vectors, got {sim}"
        );
    }

    #[test]
    fn test_orthogonal_vectors() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = CosineSimilarityF64::new(device).unwrap();

        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        let sim = op.similarity(&a, &b).unwrap();
        assert!(
            sim.abs() < 1e-10,
            "Expected 0 for orthogonal vectors, got {sim}"
        );
    }

    #[test]
    fn test_opposite_vectors() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = CosineSimilarityF64::new(device).unwrap();

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];

        let sim = op.similarity(&a, &b).unwrap();
        assert!(
            (sim + 1.0).abs() < 1e-10,
            "Expected -1.0 for opposite vectors, got {sim}"
        );
    }

    #[test]
    fn test_all_pairs() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = CosineSimilarityF64::new(device).unwrap();

        let vectors_a = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let vectors_b = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];

        let result = op.all_pairs(&vectors_a, &vectors_b, 2).unwrap();

        assert!((result[0] - 1.0).abs() < 1e-8);
        assert!(result[1].abs() < 1e-8);
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert!((result[2] - sqrt2_inv).abs() < 1e-8);
    }

    #[test]
    fn test_large_vectors() {
        let Some(device) = get_test_device() else {
            return;
        };
        let op = CosineSimilarityF64::new(device).unwrap();

        let n = 1000;
        let a: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64).cos()).collect();

        let gpu_sim = op.similarity(&a, &b).unwrap();
        let cpu_sim = op.similarity_cpu(&a, &b);

        let rel_tol = 1e-6;
        let abs_tol = 1e-10;
        let diff = (gpu_sim - cpu_sim).abs();
        let tol = rel_tol * cpu_sim.abs().max(1.0) + abs_tol;
        assert!(
            diff < tol,
            "GPU: {gpu_sim}, CPU: {cpu_sim}, diff: {diff}, tol: {tol}"
        );
    }
}
