// SPDX-License-Identifier: AGPL-3.0-only
//! `BatchPairReduceF64` — O(N²) pairwise batch reduction (f64)
//!
//! Supported operations:
//! - `DotProduct`    — Σ A[i,d]·B[j,d]
//! - `SquaredL2`     — Σ (A[i,d]-B[j,d])²
//! - `L1Distance`    — Σ |A[i,d]-B[j,d]|
//! - `LogSumExpDiff` — Σ log(A[i,d]/B[j,d])  (DADA2 E-step)
//!
//! `WetSpring` use cases: DADA2 error model, `BrayCurtis` distance matrices,
//! spectral pairwise matching.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Reduction operation for pairwise GPU computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairReduceOp {
    /// Dot product: Σ A[i,d]·B[j,d]
    DotProduct = 0,
    /// Squared L2: Σ (A[i,d]-B[j,d])²
    SquaredL2 = 1,
    /// L1 distance: Σ |A[i,d]-B[j,d]|
    L1Distance = 2,
    /// Σ log(a/b) — DADA2 error-model log-likelihood
    LogSumExpDiff = 3,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PairReduceConfig {
    n_batches: u32,
    n_a: u32,
    n_b: u32,
    n_features: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated O(N²) pairwise reduction.
///
/// # Example
/// ```ignore
/// // DADA2 E-step: pairwise log-likelihood between reads and error models
/// let reducer = BatchPairReduceF64::new(device, PairReduceOp::LogSumExpDiff);
/// // mat_a: [N × D] (reads), mat_b: [M × D] (error models)
/// let scores = reducer.compute(1, n_reads, n_models, n_features, &reads, &models)?;
/// // scores: [N × M] log-likelihoods
/// ```
pub struct BatchPairReduceF64 {
    device: Arc<WgpuDevice>,
    op: PairReduceOp,
}

impl BatchPairReduceF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/math/batch_pair_reduce_f64.wgsl")
    }

    /// Create a new pairwise reducer with the specified operation.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, op: PairReduceOp) -> Self {
        Self { device, op }
    }

    /// Compute pairwise reductions.
    ///
    /// # Arguments
    /// * `n_batches`  — number of independent A/B matrix pairs
    /// * `n_a`        — rows in A (N)
    /// * `n_b`        — rows in B (M)
    /// * `n_features` — shared feature dimension D
    /// * `mat_a`      — flat `[B × N × D]` f64 slice
    /// * `mat_b`      — flat `[B × M × D]` f64 slice
    ///
    /// # Returns
    /// Flat `[B × N × M]` f64 Vec.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(
        &self,
        n_batches: u32,
        n_a: u32,
        n_b: u32,
        n_features: u32,
        mat_a: &[f64],
        mat_b: &[f64],
    ) -> Result<Vec<f64>> {
        let dev = &self.device;
        let cfg = PairReduceConfig {
            n_batches,
            n_a,
            n_b,
            n_features,
            op: self.op as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let cfg_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PairReduce Config"),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let a_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PairReduce A"),
                contents: bytemuck::cast_slice(mat_a),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let b_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PairReduce B"),
                contents: bytemuck::cast_slice(mat_b),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_n = (n_batches * n_a * n_b) as usize;
        let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PairReduce Output"),
            size: (out_n * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PairReduce BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PairReduce BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: b_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("PairReduce"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PairReduce PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PairReduce Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("PairReduce Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PairReduce Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_a.div_ceil(16), n_b.div_ceil(16), 1);
        }
        dev.submit_and_poll(Some(encoder.finish()));

        crate::utils::read_buffer_f64(dev, &out_buf, out_n)
    }
}

fn bgl_entry(idx: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_dot_product_identity() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        // 2 vectors [1,0], [0,1]: dot = 0; [1,0]·[1,0] = 1
        let a = vec![1.0_f64, 0.0, 0.0, 1.0]; // 2 × 2
        let b = vec![1.0_f64, 0.0, 0.0, 1.0];
        let reducer = BatchPairReduceF64::new(device, PairReduceOp::DotProduct);
        let out = reducer.compute(1, 2, 2, 2, &a, &b).unwrap();
        // out[0,0] = [1,0]·[1,0] = 1; out[0,1] = [1,0]·[0,1] = 0
        assert!((out[0] - 1.0).abs() < 1e-12, "dot(e1,e1)=1, got {}", out[0]);
        assert!(out[1].abs() < 1e-12, "dot(e1,e2)=0, got {}", out[1]);
        assert!(out[2].abs() < 1e-12, "dot(e2,e1)=0, got {}", out[2]);
        assert!((out[3] - 1.0).abs() < 1e-12, "dot(e2,e2)=1, got {}", out[3]);
    }

    #[tokio::test]
    async fn test_squared_l2() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        // ||[3,4] - [0,0]||² = 25
        let a = vec![3.0_f64, 4.0];
        let b = vec![0.0_f64, 0.0];
        let reducer = BatchPairReduceF64::new(device, PairReduceOp::SquaredL2);
        let out = reducer.compute(1, 1, 1, 2, &a, &b).unwrap();
        assert!(
            (out[0] - 25.0).abs() < 1e-10,
            "squared L2 = 25, got {}",
            out[0]
        );
    }
}
