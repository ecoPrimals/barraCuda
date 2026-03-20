// SPDX-License-Identifier: AGPL-3.0-or-later

//! Batched multinomial sampling for rarefaction — GPU kernel.
//!
//! Each GPU thread runs one replicate: draws `depth` reads from a community
//! described by cumulative abundance probabilities, counting how many reads
//! land in each taxon via binary search.
//!
//! Uses xoshiro128** PRNG matching `barracuda::ops::prng_xoshiro_wgsl`.
//!
//! Provenance: groundSpring metalForge → toadStool absorption
//! Signature alignment: groundSpring V37 (`cumulative_probs` + seed)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};

/// WGSL shader source for batched multinomial sampling (f64 probabilities).
pub const WGSL_BATCHED_MULTINOMIAL_F64: &str =
    include_str!("../../shaders/bio/batched_multinomial_f64.wgsl");

/// Config for batched multinomial sampling (groundSpring V37 alignment).
#[derive(Clone, Debug, Default)]
pub struct BatchedMultinomialConfig {
    /// When true, input probabilities are already cumulative (skip prefix sum).
    /// When false, input is raw probabilities; normalization and prefix-sum are applied.
    pub cumulative_probs: bool,

    /// Optional seed for RNG. When `Some`, seeds are derived from this and the
    /// caller need not provide a seeds buffer. When `None`, caller must pass seeds.
    pub seed: Option<u64>,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    n_taxa: u32,
    depth: u32,
    n_reps: u32,
    cumulative_probs: u32,
    seed_lo: u32,
    seed_hi: u32,
    _pad: [u32; 2],
}

/// GPU-backed batched multinomial sampling for rarefaction.
pub struct BatchedMultinomialGpu {
    device: Arc<WgpuDevice>,
}

impl BatchedMultinomialGpu {
    /// Create a batched multinomial sampler.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Draw `depth` multinomial samples for each of `n_reps` replicates.
    ///
    /// `probs` holds either raw probabilities or cumulative probabilities per taxon
    /// (length `n_taxa`), depending on `config.cumulative_probs`.
    /// `seeds` holds `n_reps * 4` u32 values (xoshiro128** state per replicate).
    /// Required when `config.seed` is `None`; ignored when `config.seed` is `Some`.
    ///
    /// Returns `counts[n_reps][n_taxa]` flattened row-major.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn sample(
        &self,
        probs: &[f64],
        seeds: Option<&mut Vec<u32>>,
        depth: u32,
        n_reps: u32,
        config: BatchedMultinomialConfig,
    ) -> Result<Vec<u32>> {
        let n_taxa = probs.len();

        let cumulative_probs: Vec<f64> = if config.cumulative_probs {
            probs.to_vec()
        } else {
            let sum: f64 = probs.iter().sum();
            let scale = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            let mut cumul = Vec::with_capacity(n_taxa);
            let mut acc = 0.0_f64;
            for &p in probs {
                acc += p * scale;
                cumul.push(acc);
            }
            if !cumul.is_empty() {
                cumul[n_taxa - 1] = 1.0;
            }
            cumul
        };

        let seeds_data: Vec<u32> = if let Some(seed) = config.seed {
            (0..n_reps as usize * 4)
                .map(|i| {
                    let s = seed.wrapping_add(i as u64);
                    (s ^ (s >> 32)) as u32
                })
                .collect()
        } else {
            let s = seeds.ok_or_else(|| BarracudaError::InvalidInput {
                message: "seeds required when config.seed is None".to_string(),
            })?;
            if s.len() != n_reps as usize * 4 {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "seeds length {} must equal n_reps * 4 = {}",
                        s.len(),
                        n_reps as usize * 4
                    ),
                });
            }
            s.clone()
        };

        let d = self.device.device();

        let (seed_lo, seed_hi) = config
            .seed
            .map_or((0, 0), |s| ((s & 0xFFFF_FFFF) as u32, (s >> 32) as u32));

        let params = GpuParams {
            n_taxa: n_taxa as u32,
            depth,
            n_reps,
            cumulative_probs: config.cumulative_probs as u32,
            seed_lo,
            seed_hi,
            _pad: [0, 0],
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchedMultinomial params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cumul_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchedMultinomial cumulative"),
            contents: bytemuck::cast_slice(&cumulative_probs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let seeds_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BatchedMultinomial seeds"),
            contents: bytemuck::cast_slice(&seeds_data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let counts_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedMultinomial counts"),
            size: (n_reps as usize * n_taxa * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(&self.device, "batched_multinomial")
            .shader(WGSL_BATCHED_MULTINOMIAL_F64, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &cumul_buf)
            .storage_rw(2, &seeds_buf)
            .storage_rw(3, &counts_buf)
            .dispatch(n_reps.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        let counts = self
            .device
            .read_buffer_u32(&counts_buf, n_reps as usize * n_taxa)?;

        Ok(counts)
    }
}

/// CPU reference: multinomial sampling for rarefaction.
///
/// `cumulative_probs` must be monotonically non-decreasing, ending near 1.0.
/// Returns `counts[n_taxa]` for a single replicate.
#[must_use]
pub fn multinomial_sample_cpu(
    cumulative_probs: &[f64],
    depth: u32,
    rng: &mut impl FnMut() -> f64,
) -> Vec<u32> {
    let n_taxa = cumulative_probs.len();
    let mut counts = vec![0u32; n_taxa];
    for _ in 0..depth {
        let u = rng();
        let taxon = cumulative_probs.partition_point(|&c| c < u).min(n_taxa - 1);
        counts[taxon] += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[test]
    fn cpu_total_equals_depth() {
        let cumul = vec![0.25, 0.50, 0.75, 1.0];
        let mut counter = 0u64;
        let mut rng = || {
            counter += 1;
            (counter as f64 * 0.1234567) % 1.0
        };
        let counts = multinomial_sample_cpu(&cumul, 1000, &mut rng);
        let total: u32 = counts.iter().sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn cpu_single_taxon() {
        let cumul = vec![1.0];
        let mut rng = || 0.5;
        let counts = multinomial_sample_cpu(&cumul, 100, &mut rng);
        assert_eq!(counts, vec![100]);
    }

    #[test]
    fn cpu_deterministic_ordering() {
        let cumul = vec![0.0, 0.0, 1.0];
        let mut rng = || 0.5;
        let counts = multinomial_sample_cpu(&cumul, 50, &mut rng);
        assert_eq!(counts[0], 0);
        assert_eq!(counts[1], 0);
        assert_eq!(counts[2], 50);
    }

    #[tokio::test]
    async fn gpu_total_equals_depth() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        let Ok(gpu) = BatchedMultinomialGpu::new(device) else {
            return;
        };

        let cumul = vec![0.25, 0.50, 0.75, 1.0];
        let n_reps = 8u32;
        let depth = 500u32;
        let mut seeds: Vec<u32> = (0..n_reps * 4).map(|i| 42 + i * 7).collect();
        let config = BatchedMultinomialConfig {
            cumulative_probs: true,
            seed: None,
        };

        let counts = gpu
            .sample(&cumul, Some(&mut seeds), depth, n_reps, config)
            .unwrap();
        assert_eq!(counts.len(), n_reps as usize * cumul.len());

        for rep in 0..n_reps as usize {
            let row = &counts[rep * cumul.len()..(rep + 1) * cumul.len()];
            let total: u32 = row.iter().sum();
            assert_eq!(
                total, depth,
                "replicate {rep}: sum={total}, expected {depth}"
            );
        }
    }

    #[tokio::test]
    async fn gpu_seed_path() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        let Ok(gpu) = BatchedMultinomialGpu::new(device) else {
            return;
        };

        let cumul = vec![0.25, 0.50, 0.75, 1.0];
        let n_reps = 4u32;
        let depth = 200u32;
        let config = BatchedMultinomialConfig {
            cumulative_probs: true,
            seed: Some(12345),
        };

        let counts = gpu.sample(&cumul, None, depth, n_reps, config).unwrap();
        assert_eq!(counts.len(), n_reps as usize * cumul.len());

        for rep in 0..n_reps as usize {
            let row = &counts[rep * cumul.len()..(rep + 1) * cumul.len()];
            let total: u32 = row.iter().sum();
            assert_eq!(total, depth, "replicate {rep}: sum={total}");
        }
    }

    #[tokio::test]
    async fn gpu_raw_probs_path() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        let Ok(gpu) = BatchedMultinomialGpu::new(device) else {
            return;
        };

        let raw_probs = vec![0.25, 0.25, 0.25, 0.25];
        let n_reps = 4u32;
        let depth = 100u32;
        let mut seeds: Vec<u32> = (0..n_reps * 4).map(|i| 7 + i * 11).collect();
        let config = BatchedMultinomialConfig {
            cumulative_probs: false,
            seed: None,
        };

        let counts = gpu
            .sample(&raw_probs, Some(&mut seeds), depth, n_reps, config)
            .unwrap();
        assert_eq!(counts.len(), n_reps as usize * raw_probs.len());

        for rep in 0..n_reps as usize {
            let row = &counts[rep * raw_probs.len()..(rep + 1) * raw_probs.len()];
            let total: u32 = row.iter().sum();
            assert_eq!(total, depth, "replicate {rep}: sum={total}");
        }
    }
}
