// SPDX-License-Identifier: AGPL-3.0-or-later

//! Taxonomy Naive Bayes FC — GPU kernel (f64).
//!
//! Computes log-posterior scores for metagenomic taxonomy classification.
//! One thread per (query, taxon) pair. GEMM-like log-space accumulation:
//!   score = log_prior[taxon] + Σ log_prob[taxon, feature] for present features.
//!
//! Provenance: wetSpring metagenomics → toadStool absorption

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;

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
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl TaxonomyFcGpu {
    /// Create a taxonomy FC GPU kernel.
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TaxonomyFC BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TaxonomyFC Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader_f64(WGSL_TAXONOMY_FC, Some("TaxonomyFC Shader"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TaxonomyFC Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("taxonomy_fc"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Classify queries against a taxonomy model.
    ///
    /// `log_probs_buf`: `[n_taxa × n_features]` f64 — log emission probabilities
    /// `log_priors_buf`: `[n_taxa]` f64 — log prior probabilities
    /// `features_buf`: `[n_queries × n_features]` u32 — binary feature vectors
    /// `scores_buf`: `[n_queries × n_taxa]` f64 — output log-posterior scores
    pub fn dispatch(
        &self,
        log_probs_buf: &wgpu::Buffer,
        log_priors_buf: &wgpu::Buffer,
        features_buf: &wgpu::Buffer,
        scores_buf: &wgpu::Buffer,
        n_queries: u32,
        n_taxa: u32,
        n_features: u32,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

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

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TaxonomyFC BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: log_probs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: log_priors_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: features_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: scores_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("TaxonomyFC Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TaxonomyFC Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_queries.div_ceil(16), n_taxa.div_ceil(16), 1);
        }
        q.submit(std::iter::once(encoder.finish()));
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
