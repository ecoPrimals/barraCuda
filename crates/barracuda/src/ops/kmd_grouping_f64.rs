// SPDX-License-Identifier: AGPL-3.0-or-later
//! KmdGroupingF64 — Kendrick Mass Defect homologue grouping (f64)
//!
//! Computes `[KM, NKM, KMD]` per ion.  Ions with matching NKM and KMD within
//! a tolerance window belong to the same homologous series.
//!
//! Reference: Kendrick (1963), J. Am. Chem. Soc.
//! PFAS-specific application: CF₂ repeat unit (49.9969 Da → nominal 50).
//!
//! WetSpring Exp018: 259 Jones Lab PFAS ions grouped by CH₂ / CF₂ homology.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct KmdConfig {
    n_ions: u32,
    _pad: u32,
    exact_unit: f64,
    nominal_unit: f64,
}

/// GPU-accelerated Kendrick Mass Defect calculation.
///
/// # Common repeat units
/// | Series | Exact mass | Nominal |
/// |--------|-----------|---------|
/// | CH₂    | 14.01565  | 14      |
/// | CF₂    | 49.9969   | 50      |
/// | C₂H₄   | 28.0313   | 28      |
pub struct KmdGroupingF64 {
    device: Arc<WgpuDevice>,
    /// Exact monoisotopic mass of the repeat unit.
    pub exact_unit: f64,
    /// Nominal (integer) mass of the repeat unit.
    pub nominal_unit: f64,
}

/// Per-ion KMD result.
#[derive(Debug, Clone, Copy)]
pub struct KmdResult {
    /// Kendrick mass: m × (nominal/exact)
    pub km: f64,
    /// Nominal Kendrick mass: round(KM)
    pub nkm: f64,
    /// Kendrick mass defect: NKM − KM
    pub kmd: f64,
}

/// Common repeat units for PFAS and other homologous series.
pub mod repeat_units {
    /// CH₂ (methylene) — general hydrocarbon series
    pub const CH2: (f64, f64) = (14.01565, 14.0);
    /// CF₂ (difluoromethylene) — PFAS backbone repeat unit
    pub const CF2: (f64, f64) = (49.9969, 50.0);
    /// C₂H₄ (ethylene) — polyethylene, surfactants
    pub const C2H4: (f64, f64) = (28.0313, 28.0);
}

impl KmdGroupingF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/bio/kmd_grouping_f64.wgsl")
    }

    /// Create a KMD calculator for the given repeat unit.
    ///
    /// Use `repeat_units::CF2` for PFAS, `repeat_units::CH2` for hydrocarbons.
    pub fn new(device: Arc<WgpuDevice>, exact_unit: f64, nominal_unit: f64) -> Self {
        Self {
            device,
            exact_unit,
            nominal_unit,
        }
    }

    /// Compute `[KM, NKM, KMD]` for each ion mass.
    ///
    /// Returns one `KmdResult` per input mass.
    pub fn compute(&self, masses: &[f64]) -> Result<Vec<KmdResult>> {
        let n = masses.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let dev = &self.device;
        let cfg = KmdConfig {
            n_ions: n as u32,
            _pad: 0,
            exact_unit: self.exact_unit,
            nominal_unit: self.nominal_unit,
        };

        let cfg_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KMD Config"),
                contents: bytemuck::bytes_of(&cfg),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mass_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KMD Masses"),
                contents: bytemuck::cast_slice(masses),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KMD Output"),
            size: (n * 3 * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("KMD BGL"),
                entries: &[
                    bgl_entry(0, wgpu::BufferBindingType::Uniform),
                    bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                    bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KMD BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: cfg_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: mass_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("KmdGrouping"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("KMD PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("KMD Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("KMD Encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("KMD Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((n as u32).div_ceil(256), 1, 1);
        }
        dev.submit_and_poll(Some(encoder.finish()));

        let raw = crate::utils::read_buffer_f64(dev, &out_buf, n * 3)?;
        Ok(raw
            .chunks_exact(3)
            .map(|c| KmdResult {
                km: c[0],
                nkm: c[1],
                kmd: c[2],
            })
            .collect())
    }

    /// Group ions by homologous series.
    ///
    /// Two ions belong to the same series if `|KMD_i − KMD_j| < kmd_tol`.
    /// Returns a `Vec<usize>` of group IDs (0-indexed, CPU post-pass).
    pub fn group(&self, masses: &[f64], kmd_tol: f64) -> Result<Vec<usize>> {
        let kmd_results = self.compute(masses)?;
        let n = kmd_results.len();
        let mut groups = vec![usize::MAX; n];
        let mut next_group = 0usize;

        for i in 0..n {
            if groups[i] != usize::MAX {
                continue;
            }
            groups[i] = next_group;
            for j in (i + 1)..n {
                if (kmd_results[i].kmd - kmd_results[j].kmd).abs() < kmd_tol {
                    groups[j] = next_group;
                }
            }
            next_group += 1;
        }
        Ok(groups)
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
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_kmd_ch2_series() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Simple alkane homologues: CH4 (16.0313), C2H6 (30.0469), C3H8 (44.0626)
        // These differ by CH2 (14.01565), so should have matching KMDs.
        let (exact, nominal) = repeat_units::CH2;
        let kmd = KmdGroupingF64::new(device, exact, nominal);
        let masses = vec![16.0313_f64, 30.0469, 44.0626];
        let results = kmd.compute(&masses).unwrap();
        // KMDs should be within ~0.001 Da of each other for a true homologous series
        let kmd_0 = results[0].kmd;
        let kmd_1 = results[1].kmd;
        let kmd_2 = results[2].kmd;
        assert!(
            (kmd_0 - kmd_1).abs() < 0.01,
            "CH2 homologues should have similar KMD: {kmd_0:.4} vs {kmd_1:.4}"
        );
        assert!(
            (kmd_1 - kmd_2).abs() < 0.01,
            "CH2 homologues should have similar KMD: {kmd_1:.4} vs {kmd_2:.4}"
        );
    }

    #[tokio::test]
    async fn test_kmd_nkm_is_round() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let (exact, nominal) = repeat_units::CF2;
        let kmd = KmdGroupingF64::new(device, exact, nominal);
        let masses = vec![499.9_f64, 549.8, 599.7];
        let results = kmd.compute(&masses).unwrap();
        for r in &results {
            let diff = (r.nkm - r.nkm.round()).abs();
            assert!(diff < 1e-9, "NKM should be an integer, got {}", r.nkm);
            assert!((r.nkm - r.km - r.kmd).abs() < 1e-9, "NKM = KM + KMD");
        }
    }
}
