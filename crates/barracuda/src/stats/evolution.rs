// SPDX-License-Identifier: AGPL-3.0-only
//! Population genetics primitives.
//!
//! Provenance: groundSpring `drift.rs` / `quasispecies.rs` -> toadStool absorption (S70).

#[cfg(feature = "gpu")]
use crate::device::WgpuDevice;
#[cfg(feature = "gpu")]
use crate::device::capabilities::WORKGROUP_SIZE_1D;
#[cfg(feature = "gpu")]
use crate::device::compute_pipeline::ComputeDispatch;
#[cfg(feature = "gpu")]
use crate::error::Result;
#[cfg(feature = "gpu")]
use bytemuck::{Pod, Zeroable};
#[cfg(feature = "gpu")]
use std::sync::Arc;

#[cfg(feature = "gpu")]
const SHADER_KIMURA: &str = include_str!("../shaders/bio/kimura_fixation_f64.wgsl");

#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct KimuraParams {
    n_elements: u32,
    _pad: [u32; 3],
}

#[cfg(feature = "gpu")]
/// GPU-parallel Kimura fixation probability batch.
pub struct KimuraGpu {
    device: Arc<WgpuDevice>,
}

#[cfg(feature = "gpu")]
impl KimuraGpu {
    /// Creates a new GPU-accelerated Kimura fixation probability batch from a WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch batch Kimura fixation over (`pop_size`, selection, `initial_freq`) triplets.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dispatch(
        &self,
        pop_sizes: &[f64],
        selections: &[f64],
        freqs: &[f64],
    ) -> Result<Vec<f64>> {
        let n = pop_sizes.len();
        if n != selections.len() || n != freqs.len() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "pop_sizes, selections, freqs must have same length".to_string(),
            });
        }
        if n == 0 {
            return Ok(Vec::new());
        }

        let pop_buf = self
            .device
            .create_buffer_f64_init("kimura:pop_sizes", pop_sizes);
        let sel_buf = self
            .device
            .create_buffer_f64_init("kimura:selections", selections);
        let freq_buf = self.device.create_buffer_f64_init("kimura:freqs", freqs);
        let out_buf = self.device.create_buffer_f64(n)?;

        let params = KimuraParams {
            n_elements: n as u32,
            _pad: [0, 0, 0],
        };
        let params_buf = self.device.create_uniform_buffer("kimura:params", &params);

        let wg_count = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "kimura_fixation")
            .shader(SHADER_KIMURA, "main")
            .f64()
            .storage_read(0, &pop_buf)
            .storage_read(1, &sel_buf)
            .storage_read(2, &freq_buf)
            .storage_rw(3, &out_buf)
            .uniform(4, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        self.device.read_f64_buffer(&out_buf, n)
    }
}

/// Kimura (1962) fixation probability under selection.
///
/// `P_fix = (1 - exp(-4Ns * p0)) / (1 - exp(-4Ns))`
///
/// Returns `initial_freq` in the neutral case (`|4Ns| < 1e-10`).
///
/// # Arguments
/// * `pop_size` - Effective population size (Ne)
/// * `selection` - Selection coefficient (s), positive = advantageous
/// * `initial_freq` - Starting allele frequency p₀ ∈ [0, 1]
#[must_use]
pub fn kimura_fixation_prob(pop_size: usize, selection: f64, initial_freq: f64) -> f64 {
    let four_ns = 4.0 * pop_size as f64 * selection;
    if four_ns.abs() < 1e-10 {
        return initial_freq;
    }
    let numerator = 1.0 - (-four_ns * initial_freq).exp();
    let denominator = 1.0 - (-four_ns).exp();
    if denominator.abs() < 1e-15 {
        return initial_freq;
    }
    numerator / denominator
}

/// Eigen (1971) error threshold for a quasispecies.
///
/// `μ_c = 1 - σ^(-1/L)` where σ = master fitness, L = genome length.
///
/// Returns `None` if σ ≤ 1 (no selective advantage) or L = 0.
#[must_use]
pub fn error_threshold(master_fitness: f64, genome_length: usize) -> Option<f64> {
    if master_fitness <= 1.0 || genome_length == 0 {
        return None;
    }
    Some(1.0 - master_fitness.powf(-1.0 / genome_length as f64))
}

/// Detection power for a rare taxon at given sequencing depth.
///
/// `P(detect) = 1 - (1 - p)^D`
///
/// Useful for rarefaction analysis and sampling design.
#[must_use]
pub fn detection_power(abundance: f64, depth: u64) -> f64 {
    if abundance <= 0.0 {
        return 0.0;
    }
    if abundance >= 1.0 {
        return 1.0;
    }
    1.0 - ((1.0 - abundance).ln() * depth as f64).exp()
}

/// Minimum sequencing depth to detect a rare taxon with given power.
///
/// `D* = ⌈ln(1 - P_target) / ln(1 - p)⌉`
///
/// Returns 0 if abundance is out of (0, 1).
#[must_use]
pub fn detection_threshold(abundance: f64, target_power: f64) -> u64 {
    if abundance <= 0.0 || abundance >= 1.0 {
        return 0;
    }
    let d = (1.0 - target_power).ln() / (1.0 - abundance).ln();
    d.ceil() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kimura_neutral() {
        let p = kimura_fixation_prob(1000, 0.0, 0.01);
        assert!(
            (p - 0.01).abs() < 1e-8,
            "neutral: P_fix should equal p0, got {p}"
        );
    }

    #[test]
    fn test_kimura_beneficial() {
        let p = kimura_fixation_prob(1000, 0.01, 0.01);
        assert!(p > 0.01, "beneficial allele should have P_fix > p0");
    }

    #[test]
    fn test_kimura_deleterious() {
        let p = kimura_fixation_prob(1000, -0.01, 0.01);
        assert!(p < 0.01, "deleterious allele should have P_fix < p0");
    }

    #[test]
    fn test_kimura_fixed() {
        let p = kimura_fixation_prob(100, 0.1, 1.0);
        assert!((p - 1.0).abs() < 1e-6, "already fixed should stay fixed");
    }

    #[test]
    fn test_error_threshold() {
        let mu_c = error_threshold(10.0, 100).unwrap();
        assert!(mu_c > 0.0 && mu_c < 1.0, "μ_c={mu_c}");
    }

    #[test]
    fn test_error_threshold_invalid() {
        assert!(error_threshold(0.5, 100).is_none());
        assert!(error_threshold(10.0, 0).is_none());
    }

    #[test]
    fn test_detection_power_basic() {
        let p = detection_power(0.001, 1000);
        assert!(p > 0.5, "1000 reads at 0.1% should detect >50%");
    }

    #[test]
    fn test_detection_power_edges() {
        assert!((detection_power(0.0, 1000)).abs() < 1e-12);
        assert!((detection_power(1.0, 1) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_detection_threshold() {
        let d = detection_threshold(0.001, 0.95);
        assert!(d > 2000, "need ~3000 reads for 95% power at 0.1%");
    }
}
