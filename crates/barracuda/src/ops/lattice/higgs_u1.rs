// SPDX-License-Identifier: AGPL-3.0-or-later
//! U(1) Abelian Higgs model HMC force GPU operator.
//!
//! Computes the leapfrog half-kick to gauge and Higgs momenta for a 2D U(1)
//! gauge + complex Higgs scalar field on a periodic `nt × ns` lattice.
//!
//! # Critical — Wirtinger factor
//!
//! The shader bakes in the factor of 2 from the Wirtinger derivative:
//! `dp_higgs/dt = -2 dS/dφ†`.  Do **not** apply an extra factor of 2 host-side.
//! Missing this factor causes `|ΔH| >> 1` and ~0% HMC acceptance.
//!
//! # Usage
//!
//! ```ignore
//! let op = HiggsU1HmcForce::new(
//!     device.clone(), nt, ns, beta_pl, kappa, lambda, mu_sq, dt,
//! )?;
//! op.half_kick(&link_angles_buf, &higgs_buf, &pi_links_buf, &pi_higgs_buf)?;
//! ```
//!
//! # Validation (hotSpring v0.5.16)
//!
//! CPU reference: `lattice/hmc.rs`.  `|ΔH|/H < 1e-4` at dt=0.05 on an 8×8
//! lattice with β=2, κ=0.5, λ=1, μ²=0 (confirmed via 10k HMC trajectories).

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::complex_f64::WGSL_COMPLEX64;

const HIGGS_WG: u32 = 64;
const HIGGS_SHADER_BODY: &str = include_str!("../../shaders/lattice/higgs_u1_hmc_f64.wgsl");

/// U(1) Abelian Higgs HMC force operator (2D periodic lattice).
pub struct HiggsU1HmcForce {
    device: Arc<WgpuDevice>,
    volume: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HiggsParams {
    nt: u32,
    ns: u32,
    volume: u32,
    _pad: u32,
    // f64 fields follow — 8-byte each
    beta_pl: f64,
    kappa: f64,
    lambda: f64,
    mu_sq: f64,
    dt: f64,
    _padf0: f64,
    _padf1: f64,
    _padf2: f64,
}

impl HiggsU1HmcForce {
    /// Compile the HMC force pipeline for a `nt × ns` lattice.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        device: Arc<WgpuDevice>,
        nt: u32,
        ns: u32,
        beta_pl: f64,
        kappa: f64,
        lambda: f64,
        mu_sq: f64,
        dt: f64,
    ) -> Result<Self> {
        let volume = nt * ns;
        let src = format!("{WGSL_COMPLEX64}\n{HIGGS_SHADER_BODY}");
        // compile_shader_f64 handles exp/log patching + ILP optimizer internally
        let module = device.compile_shader_f64(&src, Some("higgs_u1_hmc"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("HiggsU1:bgl"),
                entries: &[
                    uniform_bgl(0),        // params
                    storage_bgl(1, true),  // link_angles (read)
                    storage_bgl(2, true),  // higgs        (read)
                    storage_bgl(3, false), // pi_links     (read-write)
                    storage_bgl(4, false), // pi_higgs     (read-write)
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("HiggsU1:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("HiggsU1:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("hmc_half_kick"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params_data = HiggsParams {
            nt,
            ns,
            volume,
            _pad: 0,
            beta_pl,
            kappa,
            lambda,
            mu_sq,
            dt,
            _padf0: 0.0,
            _padf1: 0.0,
            _padf2: 0.0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HiggsU1:params"),
            size: std::mem::size_of::<HiggsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            volume,
            pipeline,
            bgl,
            params,
        })
    }

    /// Update the leapfrog step size in-place (no recompile needed).
    /// `dt` is the full step size; the shader divides by 2 internally for the
    /// half-kick.
    pub fn set_dt(&self, dt: f64) {
        // HiggsParams layout: 4×u32 (16 bytes) + 4×f64 before dt
        let offset: u64 = (4 * 4 + 4 * 8) as u64; // 16 + 32 = 48
        self.device
            .queue
            .write_buffer(&self.params, offset, &dt.to_le_bytes());
    }

    /// Execute one leapfrog half-kick: `π ← π − (dt/2) · ∂S/∂q`.
    /// All four buffers must be GPU-resident STORAGE buffers:
    /// - `link_angles` — `[V × 2]` f64, gauge link angles `θ_mu(x)`
    /// - `higgs`       — `[V × 2]` f64, complex Higgs field (re, im)
    /// - `pi_links`    — `[V × 2]` f64, gauge momenta (updated in-place)
    /// - `pi_higgs`    — `[V × 2]` f64, Higgs momenta (updated in-place)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn half_kick(
        &self,
        link_angles: &wgpu::Buffer,
        higgs: &wgpu::Buffer,
        pi_links: &wgpu::Buffer,
        pi_higgs: &wgpu::Buffer,
    ) -> Result<()> {
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiggsU1:bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: link_angles.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: higgs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: pi_links.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: pi_higgs.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("HiggsU1:enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiggsU1:half_kick"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(self.volume.div_ceil(HIGGS_WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }

    /// Number of lattice sites.
    #[must_use]
    pub fn volume(&self) -> u32 {
        self.volume
    }
}

// ── BGL helpers ───────────────────────────────────────────────────────────────

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
    fn test_wirtinger_factor_in_shader() {
        let src = format!("{WGSL_COMPLEX64}\n{HIGGS_SHADER_BODY}");
        assert!(src.contains("Wirtinger"), "factor-of-2 must be documented");
        assert!(src.contains("-2.0"), "Wirtinger factor must be in code");
    }

    #[test]
    fn test_params_16byte_aligned() {
        assert_eq!(std::mem::size_of::<HiggsParams>() % 16, 0);
    }

    /// Zero fields (θ=0, φ=0) must produce zero force kick.
    /// With all gauge link angles θ=0 and Higgs field φ=0+0i:
    ///   - Plaquette contribution: Im(e^{i·0}) = 0  →  gauge force = 0
    ///   - Hopping term: φ†(x)·e^{iθ}·φ(x+μ) = 0  →  Higgs force = 0
    /// After one half-kick, all momenta must remain unchanged (zero).
    /// Tests the full WGSL path on a 2×4 (nt=2, ns=4) periodic lattice.
    #[test]
    fn test_higgs_zero_fields_zero_kick_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };

        let (nt, ns) = (2u32, 4u32);
        let volume = (nt * ns) as usize; // 8

        // All fields zero → zero gradient everywhere
        let zeros_f64 = vec![0.0f64; volume * 2]; // [V × 2] f64
        let buf_bytes = zeros_f64.len() * std::mem::size_of::<f64>();

        let make_buf = |label: &str, read_write: bool| {
            let mut usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
            if read_write {
                usage |= wgpu::BufferUsages::COPY_SRC;
            }
            let buf = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_bytes as u64,
                usage,
                mapped_at_creation: false,
            });
            device
                .queue
                .write_buffer(&buf, 0, bytemuck::cast_slice(&zeros_f64));
            buf
        };

        let link_buf = make_buf("test:links", false);
        let higgs_buf = make_buf("test:higgs", false);
        let pi_link_buf = make_buf("test:pi_links", true);
        let pi_higgs_buf = make_buf("test:pi_higgs", true);

        let op = HiggsU1HmcForce::new(
            device.clone(),
            nt,
            ns,
            2.0,  // beta_pl
            0.5,  // kappa
            1.0,  // lambda
            0.0,  // mu_sq
            0.05, // dt
        )
        .unwrap();
        op.half_kick(&link_buf, &higgs_buf, &pi_link_buf, &pi_higgs_buf)
            .unwrap();

        // Readback pi_links and pi_higgs — both must remain zero
        let readback = |src: &wgpu::Buffer| -> Vec<f64> {
            let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("test:staging"),
                size: buf_bytes as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_encoder_guarded(&Default::default());
            enc.copy_buffer_to_buffer(src, 0, &staging, 0, buf_bytes as u64);
            device.submit_and_poll(Some(enc.finish()));
            let n_f64 = buf_bytes / std::mem::size_of::<f64>();
            device.map_staging_buffer(&staging, n_f64).unwrap()
        };

        for (i, v) in readback(&pi_link_buf).iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "pi_links[{i}] = {v:.15e}, expected 0.0 for zero fields"
            );
        }
        for (i, v) in readback(&pi_higgs_buf).iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "pi_higgs[{i}] = {v:.15e}, expected 0.0 for zero fields"
            );
        }
    }
}
