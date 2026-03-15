// SPDX-License-Identifier: AGPL-3.0-only
//! `BatchToleranceSearchF64` — GPU PFAS ion batch tolerance search
//!
//! Matches S environmental sample ion masses against R PFAS library reference
//! ions in a single GPU dispatch.  Each output score is in [0, 1]: 1.0 = exact
//! match, 0.0 = outside tolerance, linearly interpolated between.
//!
//! `WetSpring` Exp018 use case: 10,000 environmental samples × 259 Jones Lab
//! reference ions = 2.59 M comparisons per batch.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TolSearchConfig {
    n_samples: u32,
    n_refs: u32,
    _pad0: u32,
    _pad1: u32,
    ppm_tol: f64,
    da_tol: f64,
}

/// GPU-accelerated PFAS ion batch tolerance search.
///
/// # Example
/// ```ignore
/// let searcher = BatchToleranceSearchF64::new(device, 5.0, 0.005);
/// // sample_masses: [S] measured m/z; ref_masses: [R] library m/z
/// let scores = searcher.search(&sample_masses, &ref_masses)?;
/// // scores: [S × R] f32 match scores in [0,1]
/// ```
pub struct BatchToleranceSearchF64 {
    device: Arc<WgpuDevice>,
    /// PPM tolerance (e.g., 5.0 for ±5 ppm).
    pub ppm_tol: f64,
    /// Absolute Da tolerance (fallback for low-mass ions).
    pub da_tol: f64,
}

impl BatchToleranceSearchF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/bio/batch_tolerance_search_f64.wgsl")
    }

    /// Create a new tolerance searcher.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, ppm_tol: f64, da_tol: f64) -> Self {
        Self {
            device,
            ppm_tol,
            da_tol,
        }
    }

    /// Search `sample_masses` against `ref_masses`.
    ///
    /// Returns flat `[S × R]` f32 Vec (row-major, sample outer index).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn search(&self, sample_masses: &[f64], ref_masses: &[f64]) -> Result<Vec<f32>> {
        let s = sample_masses.len() as u32;
        let r = ref_masses.len() as u32;
        let dev = &self.device;

        let cfg = TolSearchConfig {
            n_samples: s,
            n_refs: r,
            _pad0: 0,
            _pad1: 0,
            ppm_tol: self.ppm_tol,
            da_tol: self.da_tol,
        };

        let cfg_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TolSearch Config"),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let sample_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TolSearch Samples"),
                contents: bytemuck::cast_slice(sample_masses),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let ref_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TolSearch Refs"),
                contents: bytemuck::cast_slice(ref_masses),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_n = (s * r) as usize;
        let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TolSearch Output"),
            size: (out_n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("TolSearch BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TolSearch BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sample_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: ref_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("TolSearch"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TolSearch PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TolSearch Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("TolSearch Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TolSearch Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(s.div_ceil(16), r.div_ceil(16), 1);
        }
        dev.submit_commands(Some(encoder.finish()));

        crate::utils::read_buffer(dev, &out_buf, out_n)
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
    async fn test_exact_match_scores_one() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let searcher = BatchToleranceSearchF64::new(device, 5.0, 0.002);
        let samples = vec![100.0_f64, 200.0, 300.0];
        let refs = vec![100.0_f64, 250.0];
        let scores = searcher.search(&samples, &refs).unwrap();
        // samples[0]=100 matches refs[0]=100 exactly → score 1.0
        assert!(
            (scores[0] - 1.0).abs() < 1e-6,
            "exact match = 1.0, got {}",
            scores[0]
        );
        // samples[0]=100 vs refs[1]=250 → no match → 0.0
        assert!(scores[1].abs() < 1e-6, "no match = 0.0, got {}", scores[1]);
    }

    #[tokio::test]
    async fn test_within_ppm_tolerance() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        // 5 ppm at 200 Da = 1e-3 Da; shift by 0.5e-3 → score 0.5
        let searcher = BatchToleranceSearchF64::new(device, 5.0, 1e-6);
        let tol = 200.0 * 5e-6;
        let samples = vec![200.0 + tol * 0.5];
        let refs = vec![200.0_f64];
        let scores = searcher.search(&samples, &refs).unwrap();
        assert!(
            scores[0] > 0.4 && scores[0] < 0.6,
            "half-tol score ≈ 0.5, got {}",
            scores[0]
        );
    }
}
