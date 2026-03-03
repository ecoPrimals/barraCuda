// SPDX-License-Identifier: AGPL-3.0-only

//! TransE knowledge graph triple scoring on GPU (f64).
//!
//! `score(h, r, t) = -‖entity[h] + relation[r] - entity[t]‖₂`
//!
//! Fully parallel over triples — one thread per triple, no sparse matrices.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TranseParams {
    n_triples: u32,
    dim: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Batch TransE triple scoring on GPU.
///
/// Entities and relations are flat `[count, dim]` row-major f64 embeddings.
/// Triple indices are `(head, relation, tail)` tuples indexing into them.
pub struct TranseScoreF64<'a> {
    pub entities: &'a [f64],
    pub relations: &'a [f64],
    pub n_entities: usize,
    pub n_relations: usize,
    pub dim: usize,
    pub heads: &'a [u32],
    pub rels: &'a [u32],
    pub tails: &'a [u32],
}

impl<'a> TranseScoreF64<'a> {
    pub fn execute(&self, device: &Arc<WgpuDevice>) -> Result<Vec<f64>> {
        let n_triples = self.heads.len();
        if n_triples != self.rels.len() || n_triples != self.tails.len() {
            return Err(BarracudaError::InvalidInput {
                message: "heads, rels, tails must have same length".into(),
            });
        }
        if self.entities.len() != self.n_entities * self.dim {
            return Err(BarracudaError::InvalidShape {
                expected: vec![self.n_entities, self.dim],
                actual: vec![self.entities.len()],
            });
        }
        if self.relations.len() != self.n_relations * self.dim {
            return Err(BarracudaError::InvalidShape {
                expected: vec![self.n_relations, self.dim],
                actual: vec![self.relations.len()],
            });
        }
        if n_triples == 0 {
            return Ok(vec![]);
        }

        let ent_buf = Self::f64_buf(device, "transe:entities", self.entities);
        let rel_buf = Self::f64_buf(device, "transe:relations", self.relations);
        let head_buf = Self::u32_buf(device, "transe:heads", self.heads);
        let rel_idx_buf = Self::u32_buf(device, "transe:rels", self.rels);
        let tail_buf = Self::u32_buf(device, "transe:tails", self.tails);

        let scores_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transe:scores"),
            size: (n_triples * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params = TranseParams {
            n_triples: n_triples as u32,
            dim: self.dim as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("transe:params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(
            include_str!("../shaders/bio/transe_score_f64.wgsl"),
            Some("transe_score_f64"),
        );

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("transe BGL"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, true),
                    storage_entry(4, true),
                    storage_entry(5, false),
                    uniform_entry(6),
                ],
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("transe BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ent_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: head_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: rel_idx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tail_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scores_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("transe PL"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("transe_score_f64"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("transe"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("transe"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((n_triples as u32).div_ceil(256), 1, 1);
        }

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("transe:staging"),
            size: (n_triples * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(&scores_buf, 0, &staging, 0, (n_triples * 8) as u64);

        device.submit_and_poll(Some(encoder.finish()));

        let result: Vec<f64> = device.map_staging_buffer(&staging, n_triples)?;
        Ok(result)
    }

    fn f64_buf(device: &Arc<WgpuDevice>, label: &str, data: &[f64]) -> wgpu::Buffer {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: &bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    fn u32_buf(device: &Arc<WgpuDevice>, label: &str, data: &[u32]) -> wgpu::Buffer {
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn transe_cpu(
        entities: &[f64],
        relations: &[f64],
        dim: usize,
        heads: &[u32],
        rels: &[u32],
        tails: &[u32],
    ) -> Vec<f64> {
        heads
            .iter()
            .zip(rels.iter().zip(tails.iter()))
            .map(|(&h, (&r, &t))| {
                let mut sum_sq = 0.0;
                for d in 0..dim {
                    let diff = entities[h as usize * dim + d] + relations[r as usize * dim + d]
                        - entities[t as usize * dim + d];
                    sum_sq += diff * diff;
                }
                -sum_sq.sqrt()
            })
            .collect()
    }

    #[tokio::test]
    async fn test_transe_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let dim = 4;
        let entities = vec![
            1.0, 0.0, 0.0, 0.0, // entity 0
            0.0, 1.0, 0.0, 0.0, // entity 1
            0.0, 0.0, 1.0, 0.0, // entity 2
        ];
        let relations = vec![
            -1.0, 1.0, 0.0, 0.0, // relation 0: maps e0 → e1
        ];
        let heads = vec![0u32];
        let rels = vec![0u32];
        let tails = vec![1u32];

        let expected = transe_cpu(&entities, &relations, dim, &heads, &rels, &tails);
        let op = TranseScoreF64 {
            entities: &entities,
            relations: &relations,
            n_entities: 3,
            n_relations: 1,
            dim,
            heads: &heads,
            rels: &rels,
            tails: &tails,
        };
        let got = op.execute(&device).unwrap();

        assert_eq!(got.len(), 1);
        assert!(
            (got[0] - expected[0]).abs() < 1e-10,
            "got {}, expected {}",
            got[0],
            expected[0]
        );
    }

    #[tokio::test]
    async fn test_transe_zero_distance() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let dim = 3;
        let entities = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let relations = vec![3.0, 3.0, 3.0]; // exactly maps e0 → e1
        let heads = vec![0u32];
        let rels = vec![0u32];
        let tails = vec![1u32];

        let op = TranseScoreF64 {
            entities: &entities,
            relations: &relations,
            n_entities: 2,
            n_relations: 1,
            dim,
            heads: &heads,
            rels: &rels,
            tails: &tails,
        };
        let got = op.execute(&device).unwrap();
        assert!(
            got[0].abs() < 1e-10,
            "perfect triple should score ~0, got {}",
            got[0]
        );
    }
}
