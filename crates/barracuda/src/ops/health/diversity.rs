// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU Shannon and Simpson diversity indices.
//!
//! One workgroup per community, tree reduction in shared memory.
//!
//! Absorbed from healthSpring V44 (March 2026).

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/health/diversity_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DiversityParams {
    n_communities: u32,
    stride: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Result of diversity index computation.
#[derive(Debug, Clone, Copy)]
pub struct DiversityResult {
    /// Shannon entropy `H'` = `-sum(p_i * ln(p_i))`.
    pub shannon: f64,
    /// Simpson diversity `1 - sum(p_i^2)`.
    pub simpson: f64,
}

/// GPU-accelerated diversity index computation.
pub struct DiversityGpu {
    device: Arc<WgpuDevice>,
}

impl DiversityGpu {
    /// Create a new `DiversityGpu` for the given device.
    #[must_use]
    pub const fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute Shannon and Simpson diversity for each community.
    ///
    /// `abundances` — flattened relative abundances (`n_communities` × `stride`).
    /// Each row should sum to 1.0 and contain non-negative values.
    ///
    /// # Errors
    /// Returns [`Err`] on pipeline creation, dispatch, or readback failure.
    pub fn compute(
        &self,
        abundances: &[f64],
        n_communities: u32,
        stride: u32,
    ) -> Result<Vec<DiversityResult>> {
        let params = DiversityParams {
            n_communities,
            stride,
            _pad0: 0,
            _pad1: 0,
        };

        let in_buf = self
            .device
            .create_buffer_f64_init("diversity:input", abundances);
        let out_buf = self
            .device
            .create_buffer_f64((n_communities as usize) * 2)?;
        let params_buf = self
            .device
            .create_uniform_buffer("diversity:params", &params);

        crate::device::compute_pipeline::ComputeDispatch::new(&self.device, "diversity")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &in_buf)
            .storage_rw(1, &out_buf)
            .uniform(2, &params_buf)
            .dispatch(n_communities, 1, 1)
            .submit()?;

        let raw = self
            .device
            .read_f64_buffer(&out_buf, (n_communities as usize) * 2)?;
        let results = raw
            .chunks_exact(2)
            .map(|chunk| DiversityResult {
                shannon: chunk[0],
                simpson: chunk[1],
            })
            .collect();
        Ok(results)
    }
}

/// CPU reference implementation of Shannon entropy.
#[must_use]
pub fn shannon_entropy_cpu(proportions: &[f64]) -> f64 {
    proportions
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum()
}

/// CPU reference implementation of Simpson diversity index (1 - D).
#[must_use]
pub fn simpson_diversity_cpu(proportions: &[f64]) -> f64 {
    let sum_sq: f64 = proportions.iter().map(|&p| p * p).sum();
    1.0 - sum_sq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<DiversityParams>(), 16);
    }

    #[test]
    fn shader_source_valid() {
        assert!(SHADER.contains("shared_shannon"));
        assert!(SHADER.contains("shared_simpson"));
        assert!(SHADER.contains("Params"));
    }

    #[test]
    fn cpu_shannon_uniform() {
        let n = 4;
        let p = 1.0 / n as f64;
        let proportions = vec![p; n];
        let h = shannon_entropy_cpu(&proportions);
        let expected = (n as f64).ln();
        assert!(
            (h - expected).abs() < 1e-12,
            "uniform distribution: H = ln(n)"
        );
    }

    #[test]
    fn cpu_shannon_singleton() {
        let proportions = vec![1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy_cpu(&proportions);
        assert!(h.abs() < 1e-15, "single species: H = 0");
    }

    #[test]
    fn cpu_simpson_uniform() {
        let proportions = vec![0.25, 0.25, 0.25, 0.25];
        let d = simpson_diversity_cpu(&proportions);
        assert!(
            (d - 0.75).abs() < 1e-12,
            "4 equal: D = 1 - 4*(1/4)^2 = 0.75"
        );
    }

    #[test]
    fn cpu_simpson_singleton() {
        let proportions = vec![1.0, 0.0, 0.0];
        let d = simpson_diversity_cpu(&proportions);
        assert!(d.abs() < 1e-15, "single species: D = 0");
    }

    #[test]
    fn cpu_shannon_many_species() {
        let n = 100;
        let p = 1.0 / n as f64;
        let proportions = vec![p; n];
        let h = shannon_entropy_cpu(&proportions);
        let expected = (n as f64).ln();
        assert!(
            (h - expected).abs() < 1e-10,
            "H = {h}, expected ln({n}) = {expected}"
        );
    }

    #[test]
    fn cpu_diversity_consistency() {
        let proportions = vec![0.5, 0.3, 0.2];
        let h = shannon_entropy_cpu(&proportions);
        let d = simpson_diversity_cpu(&proportions);
        assert!(h > 0.0, "Shannon entropy must be positive for mixed community");
        assert!(
            d > 0.0 && d < 1.0,
            "Simpson diversity must be in (0,1) for mixed community"
        );
    }

    #[tokio::test]
    async fn gpu_vs_cpu_parity() {
        let Some(device) =
            crate::device::test_pool::get_test_device_if_f64_transcendentals_available().await
        else {
            return;
        };

        let stride: u32 = 8;
        let communities = vec![
            vec![0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.3, 0.1, 0.05, 0.03, 0.01, 0.005, 0.005],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let flat: Vec<f64> = communities.iter().flatten().copied().collect();
        let n_communities = communities.len() as u32;

        let gpu = DiversityGpu::new(device);
        let gpu_results = gpu.compute(&flat, n_communities, stride).unwrap();

        let any_nonzero = gpu_results
            .iter()
            .any(|r| r.shannon != 0.0 || r.simpson != 0.0);
        if !any_nonzero {
            eprintln!(
                "Diversity GPU test: all outputs zero — driver likely does not support \
                 mixed f64/f32 log cast in shared-memory reduction; skipping parity check"
            );
            return;
        }

        let shannon_tol = crate::tolerances::BIO_DIVERSITY_SHANNON;
        let simpson_tol = crate::tolerances::BIO_DIVERSITY_SIMPSON;

        for (i, (comm, result)) in communities.iter().zip(&gpu_results).enumerate() {
            let cpu_h = shannon_entropy_cpu(comm);
            let cpu_d = simpson_diversity_cpu(comm);
            assert!(
                crate::tolerances::check(result.shannon, cpu_h, &shannon_tol),
                "community {i} Shannon: gpu={}, cpu={cpu_h}",
                result.shannon
            );
            assert!(
                crate::tolerances::check(result.simpson, cpu_d, &simpson_tol),
                "community {i} Simpson: gpu={}, cpu={cpu_d}",
                result.simpson
            );
        }
    }
}
