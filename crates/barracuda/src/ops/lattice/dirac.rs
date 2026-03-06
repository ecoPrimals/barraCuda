// SPDX-License-Identifier: AGPL-3.0-or-later

//! Staggered Dirac operator for lattice QCD (GPU + CPU reference).
//!
//! The staggered (Kogut-Susskind) Dirac operator acts on a single-component
//! complex color vector at each lattice site:
//!
//!   (`D_st` ψ)(x) = m ψ(x) + (1/2) `Σ_μ` `η_μ(x)` [`U_μ(x)` ψ(x+μ) - U_μ†(x-μ) ψ(x-μ)]
//!
//! where `η_μ(x)` = (-`1)^{x_0` + ... + x_{μ-1}} are the staggered phases.
//!
//! ## GPU Strategy
//!
//! One thread per lattice site. Complex SU(3)×color multiplication with
//! staggered phase factors, f64 throughout. All topology pre-computed on CPU
//! into flat GPU-friendly arrays via [`DiracGpuLayout`].
//!
//! ## Absorbed from
//!
//! hotSpring v0.6.1 `lattice/dirac.rs` (Feb 2026) — 8/8 GPU checks,
//! tolerance 1.78e-15.
//!
//! ## References
//!
//! - Kogut & Susskind, PRD 11, 395 (1975)
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 5

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const DIRAC_WG: u32 = 64;
const DIRAC_SHADER: &str = include_str!("../../shaders/lattice/dirac_staggered_f64.wgsl");

// ─── GPU pipeline ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DiracParams {
    volume: u32,
    pad0: u32,
    mass_re: f64,
    hop_sign: f64,
}

/// Staggered Dirac operator on a 4D SU(3) lattice (GPU).
pub struct StaggeredDirac {
    device: Arc<WgpuDevice>,
    volume: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl StaggeredDirac {
    /// Compile the Dirac pipeline for the given lattice volume.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let module = device.compile_shader_f64(DIRAC_SHADER, Some("dirac_staggered"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("StaggeredDirac:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),  // links
                    storage_bgl(2, true),  // psi_in
                    storage_bgl(3, false), // psi_out
                    storage_bgl(4, true),  // nbr
                    storage_bgl(5, true),  // phases
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("StaggeredDirac:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("StaggeredDirac:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("dirac"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            volume,
            pipeline,
            bgl,
        })
    }

    /// Dispatch `D_st` on GPU-resident buffers.
    ///
    /// * `mass` — fermion mass parameter
    /// * `hop_sign` — `+1.0` for D, `-1.0` for D† (adjoint)
    /// * `links_buf` — `[V × 4 × 18]` f64 gauge links
    /// * `psi_in` — `[V × 6]` f64 input fermion field
    /// * `psi_out` — `[V × 6]` f64 output (overwritten)
    /// * `nbr_buf` — `[V × 8]` u32 neighbor table
    /// * `phases_buf` — `[V × 4]` f64 staggered phases
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dispatch(
        &self,
        mass: f64,
        hop_sign: f64,
        links_buf: &wgpu::Buffer,
        psi_in: &wgpu::Buffer,
        psi_out: &wgpu::Buffer,
        nbr_buf: &wgpu::Buffer,
        phases_buf: &wgpu::Buffer,
    ) -> Result<()> {
        let params_data = DiracParams {
            volume: self.volume,
            pad0: 0,
            mass_re: mass,
            hop_sign,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("StaggeredDirac:params"),
            size: std::mem::size_of::<DiracParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("StaggeredDirac:bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: links_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: psi_in.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: psi_out.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: nbr_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: phases_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("StaggeredDirac:enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("StaggeredDirac:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(self.volume.div_ceil(DIRAC_WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }
}

// ─── GPU data layout ─────────────────────────────────────────────────────────

/// Pre-computed GPU-friendly layout for the Dirac operator.
///
/// Flattens 4D lattice topology, gauge links, neighbor tables, and staggered
/// phases into contiguous arrays ready for GPU upload.
pub struct DiracGpuLayout {
    /// Lattice volume (product of dims)
    pub volume: usize,
    /// `[V × 4 × 18]` f64 — gauge links (SU(3) row-major, re/im interleaved)
    pub links_flat: Vec<f64>,
    /// `[V × 8]` u32 — neighbor indices (4 dirs × forward/backward)
    pub neighbors: Vec<u32>,
    /// `[V × 4]` f64 — staggered phases `η_μ(x)` = ±1.0
    pub phases: Vec<f64>,
}

impl DiracGpuLayout {
    /// Build layout for a periodic 4D lattice with given dimensions.
    ///
    /// `dims` is `[nt, nx, ny, nz]`. `links` provides the gauge field:
    /// `links[(site * 4 + mu) * 18 + row * 6 + col * 2 + 0/1]` for re/im.
    #[must_use]
    pub fn new(dims: [usize; 4], links_data: Vec<f64>) -> Self {
        let volume = dims.iter().product::<usize>();
        let mut neighbors = vec![0u32; volume * 8];
        let mut phases = vec![0.0f64; volume * 4];

        for idx in 0..volume {
            let coords = index_to_coords(idx, &dims);
            for mu in 0..4 {
                let fwd = site_neighbor(&coords, &dims, mu, true);
                let bwd = site_neighbor(&coords, &dims, mu, false);
                neighbors[idx * 8 + mu * 2] = fwd as u32;
                neighbors[idx * 8 + mu * 2 + 1] = bwd as u32;
                phases[idx * 4 + mu] = staggered_phase(&coords, mu);
            }
        }

        Self {
            volume,
            links_flat: links_data,
            neighbors,
            phases,
        }
    }
}

// ─── Lattice geometry helpers ────────────────────────────────────────────────

fn index_to_coords(idx: usize, dims: &[usize; 4]) -> [usize; 4] {
    let mut coords = [0usize; 4];
    let mut rem = idx;
    for d in (0..4).rev() {
        coords[d] = rem % dims[d];
        rem /= dims[d];
    }
    coords
}

fn coords_to_index(coords: &[usize; 4], dims: &[usize; 4]) -> usize {
    coords[0] * dims[1] * dims[2] * dims[3]
        + coords[1] * dims[2] * dims[3]
        + coords[2] * dims[3]
        + coords[3]
}

fn site_neighbor(coords: &[usize; 4], dims: &[usize; 4], mu: usize, forward: bool) -> usize {
    let mut c = *coords;
    if forward {
        c[mu] = (c[mu] + 1) % dims[mu];
    } else {
        c[mu] = (c[mu] + dims[mu] - 1) % dims[mu];
    }
    coords_to_index(&c, dims)
}

fn staggered_phase(coords: &[usize; 4], mu: usize) -> f64 {
    let sum: usize = coords.iter().take(mu).sum();
    if sum.is_multiple_of(2) { 1.0 } else { -1.0 }
}

// ─── BGL helpers ─────────────────────────────────────────────────────────────

fn storage_bgl(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

fn uniform_bgl(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

    #[test]
    fn staggered_phases_correct() {
        assert_eq!(staggered_phase(&[0, 0, 0, 0], 0), 1.0);
        assert_eq!(staggered_phase(&[1, 0, 0, 0], 1), -1.0);
        assert_eq!(staggered_phase(&[1, 1, 0, 0], 2), 1.0);
        assert_eq!(staggered_phase(&[1, 1, 1, 0], 3), -1.0);
    }

    #[test]
    fn index_roundtrip() {
        let dims = [4, 4, 4, 4];
        for idx in 0..256 {
            let c = index_to_coords(idx, &dims);
            assert_eq!(coords_to_index(&c, &dims), idx);
        }
    }

    #[test]
    fn neighbor_periodic() {
        let dims = [4, 4, 4, 4];
        let c = [3, 0, 0, 0];
        let fwd = site_neighbor(&c, &dims, 0, true);
        let fwd_c = index_to_coords(fwd, &dims);
        assert_eq!(fwd_c[0], 0, "forward wraps: 3→0");
    }

    #[test]
    fn gpu_layout_dimensions() {
        let dims = [2, 2, 2, 2];
        let vol = 16;
        let links = vec![0.0f64; vol * 4 * 18];
        let layout = DiracGpuLayout::new(dims, links);
        assert_eq!(layout.volume, 16);
        assert_eq!(layout.links_flat.len(), 16 * 4 * 18);
        assert_eq!(layout.neighbors.len(), 16 * 8);
        assert_eq!(layout.phases.len(), 16 * 4);
    }

    #[test]
    fn test_dirac_shader_compiles() {
        assert!(DIRAC_SHADER.contains("fn dirac"));
        assert!(DIRAC_SHADER.contains("mass_re"));
    }
}
