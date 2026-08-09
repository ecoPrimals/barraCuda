// SPDX-License-Identifier: AGPL-3.0-or-later

//! Taxonomy Naive Bayes FC — GPU kernel (f64).
//!
//! Computes log-posterior scores for metagenomic taxonomy classification.
//! One thread per (query, taxon) pair. GEMM-like log-space accumulation:
//!   score = `log_prior`[taxon] + Σ `log_prob`[taxon, feature] for present features.
//!
//! Provenance: wetSpring metagenomics → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;

/// WGSL shader for taxonomy naive Bayes fully-connected classification.
pub const WGSL_TAXONOMY_FC: &str = include_str!("../../shaders/bio/taxonomy_fc.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TaxConfig {
    n_queries: u32,
    n_taxa: u32,
    n_features: u32,
    _pad: u32,
}

/// GPU kernel for taxonomy naive Bayes log-posterior scores.
pub struct TaxonomyFcGpu {
    device: Arc<WgpuDevice>,
}

impl TaxonomyFcGpu {
    /// Create a taxonomy FC GPU kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Classify queries against a taxonomy model.
    ///
    /// `log_probs_buf`: `[n_taxa × n_features]` f64 — log emission probabilities
    /// `log_priors_buf`: `[n_taxa]` f64 — log prior probabilities
    /// `features_buf`: `[n_queries × n_features]` u32 — binary feature vectors
    /// `scores_buf`: `[n_queries × n_taxa]` f64 — output log-posterior scores
    /// # Errors
    ///
    /// Returns [`Err`] if shader compilation or GPU dispatch fails.
    pub fn dispatch(
        &self,
        log_probs_buf: &wgpu::Buffer,
        log_priors_buf: &wgpu::Buffer,
        features_buf: &wgpu::Buffer,
        scores_buf: &wgpu::Buffer,
        n_queries: u32,
        n_taxa: u32,
        n_features: u32,
    ) -> Result<()> {
        let d = self.device.device();

        let config = TaxConfig {
            n_queries,
            n_taxa,
            n_features,
            _pad: 0,
        };
        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TaxonomyFC Config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        ComputeDispatch::new(&self.device, "TaxonomyFC")
            .shader(WGSL_TAXONOMY_FC, "taxonomy_fc")
            .f64()
            .uniform(0, &config_buf)
            .storage_read(1, log_probs_buf)
            .storage_read(2, log_priors_buf)
            .storage_read(3, features_buf)
            .storage_rw(4, scores_buf)
            .dispatch(n_queries.div_ceil(16), n_taxa.div_ceil(16), 1)
            .submit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_contains_entry_point() {
        assert!(WGSL_TAXONOMY_FC.contains("fn taxonomy_fc"));
    }
}
