// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU HMC leapfrog integration: momentum kick, link update, momentum generation.
//!
//! Supports `Fp64Strategy::Concurrent` (silicon saturation): momentum kick and
//! link update route through DF64 shaders on the FP32 core array, while
//! momentum generation stays native f64 for RNG precision.

use crate::device::WgpuDevice;
use crate::device::capabilities::{DeviceCapabilities, Fp64Strategy};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// Per-link workgroup size for native f64 shaders.
const WG_LINK: u32 = 128;

/// Per-link workgroup size for DF64 shaders (reduced from 128 to avoid
/// VGPR exhaustion on RDNA2: the link update's matrix inversion requires
/// ~200+ VGPRs per thread; at WG128 that's 4 wave32s × 256 VGPRs = 1024,
/// exactly saturating the SIMD32 register file with zero spillroom).
const WG_LINK_DF64: u32 = 64;

/// Split workgroup count into (x, y) for 2D dispatch when total exceeds 65535.
/// Shaders must linearize: `idx = gid.y * (num_wgs.x * WG) + gid.x`.
const fn split_workgroups(total: u32) -> (u32, u32) {
    if total <= 65535 {
        (total, 1)
    } else {
        let y = (total + 65534) / 65535; // div_ceil not const-stable
        let x = (total + y - 1) / y;
        (x, y)
    }
}

use super::su3::su3_df64_preamble;
use super::su3_extended::su3_extended_preamble;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/hmc_leapfrog_f64.wgsl");
const SHADER_MOMENTUM_DF64: &str =
    include_str!("../../shaders/lattice/su3_momentum_update_df64.wgsl");
const SHADER_LINK_DF64: &str = include_str!("../../shaders/lattice/su3_link_update_df64.wgsl");

/// Uniform buffer layout for leapfrog dispatch (volume, n_links, dt).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeapfrogParams {
    /// Total lattice volume (nt × nx × ny × nz).
    pub volume: u32,
    /// Total gauge links (volume × 4).
    pub n_links: u32,
    /// Padding for 16-byte alignment.
    pub _pad0: u32,
    /// Padding for 16-byte alignment.
    pub _pad1: u32,
    /// Time step (momentum kick or link update magnitude).
    pub dt: f64,
    /// Padding for 32-byte struct alignment.
    pub _padf: f64,
}

/// GPU-resident buffers for HMC leapfrog integration steps.
pub struct LeapfrogBuffers<'a> {
    /// Gauge link field.
    pub links_buf: &'a wgpu::Buffer,
    /// Conjugate momenta field.
    pub momenta_buf: &'a wgpu::Buffer,
    /// Force (∂S/∂U) field.
    pub force_buf: &'a wgpu::Buffer,
    /// PRNG state for momentum generation.
    pub rng_buf: &'a wgpu::Buffer,
}

/// GPU HMC leapfrog integrator with strategy-aware dispatch.
///
/// In `Concurrent` mode, momentum kick and link update use DF64 shaders
/// (throughput-bound work on FP32 cores), while momentum generation uses
/// native f64 (precision-critical RNG).
///
/// When TMU tables are attached (`with_tmu`), momentum generation offloads
/// Box-Muller transcendentals (log, cos, sin) to TMU texture lookups, freeing
/// ALU for concurrent physics prep.
pub struct GpuHmcLeapfrog {
    device: Arc<WgpuDevice>,
    n_links: u32,
    native_shader_src: String,
    df64_momentum_src: Option<String>,
    df64_link_src: Option<String>,
    strategy: Fp64Strategy,
    tmu_tables: Option<super::tmu_tables::TmuLookupTables>,
}

impl GpuHmcLeapfrog {
    /// Create HMC leapfrog integrator for given lattice volume.
    ///
    /// Selects shader paths based on `Fp64Strategy`:
    /// - `Native`/`Sovereign`: all paths use native f64
    /// - `Concurrent`: momentum/link use DF64, momentum gen stays native f64
    /// - `Hybrid`: all paths use DF64
    /// # Errors
    /// Returns [`Err`] if shader compilation fails, pipeline creation fails, or the device is lost.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let n_links = volume * 4;
        let caps = DeviceCapabilities::from_device(&device);
        let strategy = caps.fp64_strategy();

        let native_shader_src = format!("{}{}", su3_extended_preamble(), SHADER_BODY);

        let (df64_momentum_src, df64_link_src) = match strategy {
            Fp64Strategy::Concurrent | Fp64Strategy::Hybrid => {
                let mom = format!("{}{}", su3_df64_preamble(), SHADER_MOMENTUM_DF64);
                let link = format!("{}{}", su3_df64_preamble(), SHADER_LINK_DF64);
                (Some(mom), Some(link))
            }
            _ => (None, None),
        };

        tracing::info!(
            ?strategy,
            "GpuHmcLeapfrog: compiled with {:?} FP64 strategy",
            strategy
        );

        Ok(Self {
            device,
            n_links,
            native_shader_src,
            df64_momentum_src,
            df64_link_src,
            strategy,
            tmu_tables: None,
        })
    }

    /// Attach TMU lookup tables for texture-accelerated momentum generation.
    ///
    /// When set, `generate_momenta` uses TMU texture lookups for Box-Muller
    /// transcendentals instead of ALU, lighting up otherwise-idle TMU units.
    #[must_use]
    pub fn with_tmu(mut self, tables: super::tmu_tables::TmuLookupTables) -> Self {
        self.tmu_tables = Some(tables);
        self
    }

    /// π ← π + dt × force
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn momentum_kick(&self, buffers: &LeapfrogBuffers<'_>, volume: u32, dt: f64) -> Result<()> {
        if let Some(ref df64_src) = self.df64_momentum_src {
            self.dispatch_df64(buffers, volume, dt, df64_src, "momentum_update_df64", "kick_df64")
        } else {
            self.dispatch_native(buffers, volume, dt, "momentum_kick", "kick")
        }
    }

    /// U ← exp(dt × π) × U  then reunitarize
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn link_update(&self, buffers: &LeapfrogBuffers<'_>, volume: u32, dt: f64) -> Result<()> {
        if let Some(ref df64_src) = self.df64_link_src {
            self.dispatch_df64(buffers, volume, dt, df64_src, "link_update_df64", "update_df64")
        } else {
            self.dispatch_native(buffers, volume, dt, "link_update", "update")
        }
    }

    /// Generate random su(3) algebra momenta.
    ///
    /// If TMU tables are attached, uses texture-accelerated Box-Muller (offloads
    /// log/cos/sin to TMU units). Otherwise falls back to native ALU path.
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn generate_momenta(&self, buffers: &LeapfrogBuffers<'_>, volume: u32) -> Result<()> {
        if let Some(ref tables) = self.tmu_tables {
            self.dispatch_tmu_momenta(buffers, volume, tables)
        } else {
            self.dispatch_native(buffers, volume, 0.0, "generate_momenta", "gen")
        }
    }

    fn dispatch_tmu_momenta(
        &self,
        buffers: &LeapfrogBuffers<'_>,
        _volume: u32,
        tables: &super::tmu_tables::TmuLookupTables,
    ) -> Result<()> {
        use super::absorbed_shaders::{WGSL_PRNG_PCG_F64, WGSL_SU3_RANDOM_MOMENTA_TMU_F64};

        let shader_src = format!("{WGSL_PRNG_PCG_F64}\n{WGSL_SU3_RANDOM_MOMENTA_TMU_F64}");

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct TmuPrngParams {
            n_links: u32,
            traj_id: u32,
            seed_lo: u32,
            seed_hi: u32,
        }

        let seed: u64 = buffers.rng_buf.size();
        let params_data = TmuPrngParams {
            n_links: self.n_links,
            traj_id: 0,
            seed_lo: seed as u32,
            seed_hi: (seed >> 32) as u32,
        };
        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tmu_prng_params"),
            size: std::mem::size_of::<TmuPrngParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

        let total_wgs = self.n_links.div_ceil(64);
        let (wx, wy) = split_workgroups(total_wgs);

        let module = self.device.compile_shader_f64(&shader_src, Some("tmu_momenta"));

        let bgl = self.device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("tmu_prng_bgl"),
                entries: &[
                    crate::device::compute_pipeline::uniform_bgl_entry(0),
                    crate::device::compute_pipeline::storage_bgl_entry(1, false),
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                ],
            },
        );

        let bg = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tmu_prng_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffers.momenta_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&tables.log_table),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&tables.trig_table),
                },
            ],
        });

        let pl = self.device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("tmu_prng_pl"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            },
        );
        let pipeline = self.device.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("tmu_prng_pipeline"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            },
        );

        let _permit = self.device.acquire_dispatch();
        let mut enc = self.device.create_encoder_guarded(
            &wgpu::CommandEncoderDescriptor {
                label: Some("tmu_momenta"),
            },
        );
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tmu_momenta_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wx, wy, 1);
        }
        self.device.submit_and_poll_inner(Some(enc.finish()));
        Ok(())
    }

    fn dispatch_native(
        &self,
        buffers: &LeapfrogBuffers<'_>,
        volume: u32,
        dt: f64,
        entry_point: &str,
        label: &str,
    ) -> Result<()> {
        self.dispatch_shader(&self.native_shader_src, buffers, volume, dt, entry_point, label, true)
    }

    fn dispatch_df64(
        &self,
        buffers: &LeapfrogBuffers<'_>,
        _volume: u32,
        dt: f64,
        shader_src: &str,
        entry_point: &str,
        label: &str,
    ) -> Result<()> {
        let params_data = LeapfrogParams {
            volume: _volume,
            n_links: self.n_links,
            _pad0: 0,
            _pad1: 0,
            dt,
            _padf: 0.0,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("GpuHmcLeapfrog:{label}:params")),
            size: std::mem::size_of::<LeapfrogParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        let total_wgs = self.n_links.div_ceil(WG_LINK_DF64);
        let (wx, wy) = split_workgroups(total_wgs);

        let dispatch_label = format!("GpuHmcLeapfrog:{label}");
        let mut dispatch = ComputeDispatch::new(&self.device, &dispatch_label);
        dispatch = dispatch.shader(shader_src, entry_point);
        dispatch
            .uniform(0, &params)
            .storage_rw(1, buffers.links_buf)
            .storage_rw(2, buffers.momenta_buf)
            .storage_read(3, buffers.force_buf)
            .storage_rw(4, buffers.rng_buf)
            .dispatch(wx, wy, 1)
            .submit()?;

        Ok(())
    }

    fn dispatch_shader(
        &self,
        shader_src: &str,
        buffers: &LeapfrogBuffers<'_>,
        volume: u32,
        dt: f64,
        entry_point: &str,
        label: &str,
        use_f64: bool,
    ) -> Result<()> {
        let params_data = LeapfrogParams {
            volume,
            n_links: self.n_links,
            _pad0: 0,
            _pad1: 0,
            dt,
            _padf: 0.0,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("GpuHmcLeapfrog:{label}:params")),
            size: std::mem::size_of::<LeapfrogParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        let dispatch_label = format!("GpuHmcLeapfrog:{label}");
        let mut dispatch = ComputeDispatch::new(&self.device, &dispatch_label);
        dispatch = dispatch.shader(shader_src, entry_point);
        if use_f64 {
            dispatch = dispatch.f64();
        }
        dispatch
            .uniform(0, &params)
            .storage_rw(1, buffers.links_buf)
            .storage_rw(2, buffers.momenta_buf)
            .storage_read(3, buffers.force_buf)
            .storage_rw(4, buffers.rng_buf)
            .dispatch(self.n_links.div_ceil(WG_LINK), 1, 1)
            .submit()?;

        Ok(())
    }

    /// Number of gauge links (volume × 4).
    #[must_use]
    pub fn n_links(&self) -> u32 {
        self.n_links
    }

    /// Active FP64 strategy used by this integrator.
    #[must_use]
    pub fn strategy(&self) -> Fp64Strategy {
        self.strategy
    }

    /// Native f64 shader source (for streaming pipeline pre-compilation).
    #[must_use]
    pub fn native_shader_src(&self) -> &str {
        &self.native_shader_src
    }

    /// DF64 momentum shader source, if Concurrent/Hybrid strategy is active.
    #[must_use]
    pub fn df64_momentum_src(&self) -> Option<&str> {
        self.df64_momentum_src.as_deref()
    }

    /// DF64 link shader source, if Concurrent/Hybrid strategy is active.
    #[must_use]
    pub fn df64_link_src(&self) -> Option<&str> {
        self.df64_link_src.as_deref()
    }

    /// Workgroup count for native dispatching over all links.
    #[must_use]
    pub fn workgroup_count(&self) -> u32 {
        self.n_links.div_ceil(WG_LINK)
    }

    /// Workgroup count for DF64 dispatching (smaller WG for register pressure).
    #[must_use]
    pub fn workgroup_count_df64(&self) -> u32 {
        self.n_links.div_ceil(WG_LINK_DF64)
    }

    /// Access the device (for streaming encoder compilation).
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leapfrog_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuHmcLeapfrog::new(device, 16).unwrap();
        assert_eq!(op.n_links(), 64);
    }
}
