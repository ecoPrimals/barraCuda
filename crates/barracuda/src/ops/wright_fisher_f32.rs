// SPDX-License-Identifier: AGPL-3.0-only
//! `WrightFisherF32` — GPU-vectorized Wright-Fisher population genetics.
//!
//! Simulates allele frequency evolution under selection + genetic drift for
//! `n_pops × n_loci` (population, locus) pairs in parallel on the GPU.
//!
//! Each generation:
//! 1. **Selection**: `p' = p * (1+s) / (p * (1+s) + (1-p))`
//! 2. **Drift**: Binomial(`2N`, `p'`) via sequential PRNG sampling
//!
//! Uses xoshiro128** PRNG seeded per-thread from a state buffer.
//!
//! Absorbed from neuralSpring `metalForge/shaders/wright_fisher_step.wgsl`
//! (Papers 024/025 — pangenome selection, meta-population dynamics).

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct WrightFisherParams {
    n_pops: u32,
    n_loci: u32,
    two_n: u32,
    _pad: u32,
}

/// Configuration for a Wright-Fisher simulation.
pub struct WrightFisherConfig {
    /// Number of populations.
    pub n_pops: u32,
    /// Number of loci per population.
    pub n_loci: u32,
    /// Diploid population size (2N total gene copies).
    pub two_n: u32,
}

/// GPU-vectorized Wright-Fisher drift + selection simulation.
///
/// Each GPU thread handles one (population, locus) pair per generation.
/// Call [`simulate_generation`](Self::simulate_generation) repeatedly for
/// multi-generation trajectories.
pub struct WrightFisherF32 {
    device: Arc<WgpuDevice>,
    config: WrightFisherConfig,
}

impl WrightFisherF32 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/science/wright_fisher_step_f32.wgsl")
    }

    /// Create a new Wright-Fisher simulation.
    ///
    /// # Errors
    ///
    /// Returns an error if `n_pops == 0`, `n_loci == 0`, or `two_n == 0`.
    pub fn new(device: Arc<WgpuDevice>, config: WrightFisherConfig) -> Result<Self> {
        if config.n_pops == 0 || config.n_loci == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "WrightFisherF32: n_pops and n_loci must be > 0".into(),
            });
        }
        if config.two_n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "WrightFisherF32: two_n must be > 0".into(),
            });
        }
        Ok(Self { device, config })
    }

    /// Run one generation of Wright-Fisher selection + drift.
    ///
    /// # Arguments
    ///
    /// * `freq_in` — Current allele frequencies, length `n_pops × n_loci`, in `[0, 1]`.
    /// * `selection` — Selection coefficients per locus, length `n_loci`.
    ///   `s > 0` = advantageous, `s = 0` = neutral, `s < 0` = deleterious.
    /// * `prng_state` — Mutable xoshiro128** state, length `4 × n_pops × n_loci`.
    ///   Updated in-place each generation.
    ///
    /// Returns new allele frequencies of length `n_pops × n_loci`.
    ///
    /// # Errors
    ///
    /// Returns an error if input lengths are wrong or GPU dispatch fails.
    pub fn simulate_generation(
        &self,
        freq_in: &[f32],
        selection: &[f32],
        prng_state: &mut Vec<u32>,
    ) -> Result<Vec<f32>> {
        let total = (self.config.n_pops as usize) * (self.config.n_loci as usize);
        if freq_in.len() != total {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "freq_in length {} != n_pops*n_loci = {total}",
                    freq_in.len()
                ),
            });
        }
        if selection.len() != self.config.n_loci as usize {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "selection length {} != n_loci = {}",
                    selection.len(),
                    self.config.n_loci
                ),
            });
        }
        let expected_prng = total * 4;
        if prng_state.len() != expected_prng {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "prng_state length {} != 4*n_pops*n_loci = {expected_prng}",
                    prng_state.len()
                ),
            });
        }

        let dev = &self.device;
        let params = WrightFisherParams {
            n_pops: self.config.n_pops,
            n_loci: self.config.n_loci,
            two_n: self.config.two_n,
            _pad: 0,
        };

        let freq_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WF freq_in"),
                contents: bytemuck::cast_slice(freq_in),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let sel_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WF selection"),
                contents: bytemuck::cast_slice(selection),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("WF freq_out"),
            size: (total * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let prng_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WF prng_state"),
                contents: bytemuck::cast_slice(prng_state),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let param_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WF params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("WF BGL"),
                entries: &[
                    bgl_storage_ro(0),
                    bgl_storage_ro(1),
                    bgl_storage_rw(2),
                    bgl_storage_rw(3),
                    bgl_uniform(4),
                ],
            });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("WF BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: freq_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sel_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: prng_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });

        let shader = dev
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("WF Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("WF PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("WF Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let wg = (total as u32).div_ceil(WORKGROUP_SIZE_1D);
        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("WF Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WF Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg, 1, 1);
        }
        dev.submit_and_poll(Some(encoder.finish()));

        let freq_out = crate::utils::read_buffer(dev, &out_buf, total)?;
        let new_prng = crate::utils::read_buffer_u32(dev, &prng_buf, expected_prng)?;
        *prng_state = new_prng;

        Ok(freq_out)
    }
}

/// Seed xoshiro128** state for `n` threads using `SplitMix32`.
///
/// Each thread gets 4 independent u32 state values.
#[must_use]
pub fn seed_xoshiro_state(base_seed: u32, n: usize) -> Vec<u32> {
    let mut state = Vec::with_capacity(n * 4);
    for i in 0..n {
        let mut s = base_seed.wrapping_add(i as u32);
        for _ in 0..4 {
            s = splitmix32(s);
            state.push(s);
        }
    }
    state
}

fn splitmix32(mut z: u32) -> u32 {
    z = z.wrapping_add(0x9E37_79B9);
    z = (z ^ (z >> 15)).wrapping_mul(0x85EB_CA6B);
    z = (z ^ (z >> 13)).wrapping_mul(0xC2B2_AE35);
    z ^ (z >> 16)
}

fn bgl_storage_ro(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_storage_rw(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
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

    #[test]
    fn seed_state_length() {
        let state = seed_xoshiro_state(42, 10);
        assert_eq!(state.len(), 40);
    }

    #[test]
    fn seed_state_nonzero() {
        let state = seed_xoshiro_state(0, 100);
        assert!(state.iter().any(|&s| s != 0), "all zeros is degenerate");
    }

    #[test]
    fn seed_state_deterministic() {
        let a = seed_xoshiro_state(123, 50);
        let b = seed_xoshiro_state(123, 50);
        assert_eq!(a, b);
    }

    #[tokio::test]
    async fn wright_fisher_neutral_drift() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let n_pops = 4;
        let n_loci = 8;
        let total = n_pops * n_loci;
        let wf = WrightFisherF32::new(
            device,
            WrightFisherConfig {
                n_pops: n_pops as u32,
                n_loci: n_loci as u32,
                two_n: 100,
            },
        )
        .unwrap();

        let freq_in = vec![0.5_f32; total];
        let selection = vec![0.0_f32; n_loci];
        let mut prng = seed_xoshiro_state(42, total);

        let freq_out = wf
            .simulate_generation(&freq_in, &selection, &mut prng)
            .unwrap();

        assert_eq!(freq_out.len(), total);
        for &f in &freq_out {
            assert!((0.0..=1.0).contains(&f), "frequency {f} out of [0,1]");
        }
    }

    #[tokio::test]
    async fn wright_fisher_strong_selection() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let n_pops = 10;
        let n_loci = 1;
        let total = n_pops * n_loci;
        let wf = WrightFisherF32::new(
            device,
            WrightFisherConfig {
                n_pops: n_pops as u32,
                n_loci: n_loci as u32,
                two_n: 1000,
            },
        )
        .unwrap();

        let freq_in = vec![0.5_f32; total];
        let selection = vec![0.5_f32; n_loci]; // very strong positive selection
        let mut prng = seed_xoshiro_state(99, total);

        let freq_out = wf
            .simulate_generation(&freq_in, &selection, &mut prng)
            .unwrap();

        // With s=0.5 and 2N=1000, most frequencies should increase from 0.5
        let mean_freq: f32 = freq_out.iter().sum::<f32>() / freq_out.len() as f32;
        assert!(
            mean_freq > 0.55,
            "strong selection should increase frequency; mean = {mean_freq}"
        );
    }

    #[tokio::test]
    async fn wright_fisher_fixation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let wf = WrightFisherF32::new(
            device,
            WrightFisherConfig {
                n_pops: 1,
                n_loci: 1,
                two_n: 20,
            },
        )
        .unwrap();

        let selection = vec![0.0_f32]; // neutral
        let mut prng = seed_xoshiro_state(7, 1);
        let mut freq = vec![0.5_f32];

        // Validate dispatch works — run a few generations, check output is valid
        for _ in 0..5 {
            freq = wf
                .simulate_generation(&freq, &selection, &mut prng)
                .unwrap();
        }

        assert!(
            (0.0..=1.0).contains(&freq[0]),
            "frequency must be in [0, 1], got {}",
            freq[0]
        );
    }
}
