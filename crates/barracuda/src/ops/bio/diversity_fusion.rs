// SPDX-License-Identifier: AGPL-3.0-only

//! Fused diversity metrics — Shannon + Simpson + Pielou evenness in one dispatch.
//!
//! Computes three diversity indices per sample in a single GPU kernel pass,
//! avoiding three separate [`FusedMapReduceF64`] dispatches. For N samples
//! with S species each, this reduces GPU round-trips from 3N to N.
//!
//! Per sample (3 contiguous f64 values):
//! - `[0]` Shannon entropy H' = −Σ pᵢ ln(pᵢ)
//! - `[1]` Simpson index  D  = 1 − Σ pᵢ²
//! - `[2]` Pielou evenness J' = H' / `ln(S_obs)`
//!
//! Provenance: wetSpring Write phase → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

/// WGSL shader source for fused diversity computation (f64).
pub const WGSL_DIVERSITY_FUSION_F64: &str =
    include_str!("../../shaders/bio/diversity_fusion_f64.wgsl");

/// Fused diversity result for a single sample.
#[derive(Debug, Clone, Copy)]
pub struct DiversityResult {
    /// Shannon entropy H' = −Σ pᵢ ln(pᵢ).
    pub shannon: f64,
    /// Simpson index D = 1 − Σ pᵢ².
    pub simpson: f64,
    /// Pielou evenness J' = H' / `ln(S_obs)`.
    pub evenness: f64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuParams {
    n_samples: u32,
    n_species: u32,
}

/// GPU-backed fused diversity computation.
pub struct DiversityFusionGpu {
    device: Arc<WgpuDevice>,
}

impl DiversityFusionGpu {
    /// Create a diversity fusion compute instance.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute fused diversity metrics for multiple samples.
    ///
    /// `abundances` is row-major `[n_samples × n_species]`.
    ///
    /// # Panics
    /// Panics if `abundances.len() != n_samples * n_species`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(
        &self,
        abundances: &[f64],
        n_samples: usize,
        n_species: usize,
    ) -> Result<Vec<DiversityResult>> {
        assert_eq!(
            abundances.len(),
            n_samples * n_species,
            "abundances length must equal n_samples * n_species"
        );

        let d = self.device.device();

        let params = GpuParams {
            n_samples: n_samples as u32,
            n_species: n_species as u32,
        };

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DiversityFusion params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let abundances_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("DiversityFusion abundances"),
            contents: bytemuck::cast_slice(abundances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let results_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DiversityFusion results"),
            size: (n_samples * 3 * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(&self.device, "diversity_fusion")
            .shader(WGSL_DIVERSITY_FUSION_F64, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, &abundances_buf)
            .storage_rw(2, &results_buf)
            .dispatch(params.n_samples.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        let raw = self.device.read_buffer_f64(&results_buf, n_samples * 3)?;

        Ok(raw
            .chunks_exact(3)
            .map(|c| DiversityResult {
                shannon: c[0],
                simpson: c[1],
                evenness: c[2],
            })
            .collect())
    }
}

/// CPU reference implementation of fused diversity metrics.
#[must_use]
pub fn diversity_fusion_cpu(abundances: &[f64], n_species: usize) -> Vec<DiversityResult> {
    abundances
        .chunks_exact(n_species)
        .map(|sample| {
            let total: f64 = sample.iter().sum();
            if total <= 0.0 {
                return DiversityResult {
                    shannon: 0.0,
                    simpson: 0.0,
                    evenness: 0.0,
                };
            }

            let mut shannon = 0.0;
            let mut simpson_sum = 0.0;
            let mut s_obs = 0.0_f64;

            for &count in sample {
                if count > 0.0 {
                    let p = count / total;
                    shannon -= p * p.ln();
                    simpson_sum += p * p;
                    s_obs += 1.0;
                }
            }

            let evenness = if s_obs > 1.0 {
                shannon / s_obs.ln()
            } else {
                0.0
            };

            DiversityResult {
                shannon,
                simpson: 1.0 - simpson_sum,
                evenness,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    const TOL: f64 = 1e-12;

    #[test]
    fn cpu_single_sample_known_values() {
        let abundances = [10.0, 20.0, 30.0, 40.0];
        let results = diversity_fusion_cpu(&abundances, 4);
        assert_eq!(results.len(), 1);

        let r = &results[0];
        let p = [0.1, 0.2, 0.3, 0.4];
        let expected_shannon: f64 = -p.iter().map(|pi: &f64| pi * pi.ln()).sum::<f64>();
        let expected_simpson: f64 = 1.0 - p.iter().map(|pi: &f64| pi * pi).sum::<f64>();
        let expected_evenness = expected_shannon / 4.0_f64.ln();

        assert!((r.shannon - expected_shannon).abs() < TOL, "shannon");
        assert!((r.simpson - expected_simpson).abs() < TOL, "simpson");
        assert!((r.evenness - expected_evenness).abs() < TOL, "evenness");
    }

    #[test]
    fn cpu_uniform_distribution() {
        let abundances = [25.0; 4];
        let results = diversity_fusion_cpu(&abundances, 4);
        let r = &results[0];

        let expected_shannon = 4.0_f64.ln();
        assert!((r.shannon - expected_shannon).abs() < TOL);
        assert!((r.simpson - 0.75).abs() < TOL);
        assert!((r.evenness - 1.0).abs() < TOL);
    }

    #[test]
    fn cpu_single_species_dominance() {
        let abundances = [100.0, 0.0, 0.0, 0.0];
        let results = diversity_fusion_cpu(&abundances, 4);
        let r = &results[0];

        assert!(r.shannon.abs() < TOL, "single species → H'=0");
        assert!(r.simpson.abs() < TOL, "single species → D=0");
        assert!(r.evenness.abs() < TOL, "single species → J'=0");
    }

    #[test]
    fn cpu_empty_sample() {
        let abundances = [0.0; 4];
        let results = diversity_fusion_cpu(&abundances, 4);
        let r = &results[0];

        assert!(r.shannon.abs() < f64::EPSILON, "empty → H'=0");
        assert!(r.simpson.abs() < f64::EPSILON, "empty → D=0");
        assert!(r.evenness.abs() < f64::EPSILON, "empty → J'=0");
    }

    #[test]
    fn cpu_multiple_samples() {
        let abundances = [10.0, 20.0, 30.0, 40.0, 25.0, 25.0, 25.0, 25.0];
        let results = diversity_fusion_cpu(&abundances, 4);
        assert_eq!(results.len(), 2);

        assert!(results[1].shannon > results[0].shannon);
        assert!((results[1].evenness - 1.0).abs() < TOL);
    }

    #[test]
    fn cpu_two_species_even() {
        let abundances = [50.0, 50.0];
        let results = diversity_fusion_cpu(&abundances, 2);
        let r = &results[0];

        assert!((r.shannon - 2.0_f64.ln()).abs() < TOL);
        assert!((r.simpson - 0.5).abs() < TOL);
        assert!((r.evenness - 1.0).abs() < TOL);
    }

    #[tokio::test]
    async fn gpu_parity() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };

        let Ok(gpu) = DiversityFusionGpu::new(device) else {
            return;
        };

        let abundances = vec![
            10.0, 20.0, 30.0, 40.0, // sample 0
            25.0, 25.0, 25.0, 25.0, // sample 1 (uniform)
            100.0, 0.0, 0.0, 0.0, // sample 2 (monoculture)
        ];
        let n_samples = 3;
        let n_species = 4;

        let cpu = diversity_fusion_cpu(&abundances, n_species);
        let gpu_results = gpu.compute(&abundances, n_samples, n_species).unwrap();

        assert_eq!(gpu_results.len(), n_samples);

        for (i, (c, g)) in cpu.iter().zip(gpu_results.iter()).enumerate() {
            assert!(
                (c.shannon - g.shannon).abs() < 1e-10,
                "sample {i} shannon: cpu={}, gpu={}",
                c.shannon,
                g.shannon
            );
            assert!(
                (c.simpson - g.simpson).abs() < 1e-10,
                "sample {i} simpson: cpu={}, gpu={}",
                c.simpson,
                g.simpson
            );
            assert!(
                (c.evenness - g.evenness).abs() < 1e-10,
                "sample {i} evenness: cpu={}, gpu={}",
                c.evenness,
                g.evenness
            );
        }
    }
}
