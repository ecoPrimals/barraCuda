// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU HMC leapfrog integration: momentum kick, link update, momentum generation.

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::su3_extended::su3_extended_preamble;

const WG: u32 = 64;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/hmc_leapfrog_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LeapfrogParams {
    volume: u32,
    n_links: u32,
    _pad0: u32,
    _pad1: u32,
    dt: f64,
    _padf: f64,
}

/// GPU HMC leapfrog integrator with three dispatch modes.
pub struct GpuHmcLeapfrog {
    device: Arc<WgpuDevice>,
    n_links: u32,
    kick_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    gen_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuHmcLeapfrog {
    /// Create HMC leapfrog integrator for given lattice volume.
    /// # Errors
    /// Returns [`Err`] if shader compilation fails, pipeline creation fails, or the device is lost.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let n_links = volume * 4;
        let src = format!("{}{}", su3_extended_preamble(), SHADER_BODY);
        let module = device.compile_shader_f64(&src, Some("hmc_leapfrog"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuHmcLeapfrog:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, false), // links (read_write)
                    storage_bgl(2, false), // momenta (read_write)
                    storage_bgl(3, true),  // force (read)
                    storage_bgl(4, false), // rng_state (read_write)
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuHmcLeapfrog:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let make_pipeline = |entry: &str, label: &str| {
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let kick_pipeline = make_pipeline("momentum_kick", "GpuHmcLeapfrog:kick");
        let update_pipeline = make_pipeline("link_update", "GpuHmcLeapfrog:update");
        let gen_pipeline = make_pipeline("generate_momenta", "GpuHmcLeapfrog:gen");

        Ok(Self {
            device,
            n_links,
            kick_pipeline,
            update_pipeline,
            gen_pipeline,
            bgl,
        })
    }

    /// π ← π + dt × force
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn momentum_kick(
        &self,
        links_buf: &wgpu::Buffer,
        momenta_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        dt: f64,
    ) -> Result<()> {
        self.dispatch(
            &self.kick_pipeline,
            links_buf,
            momenta_buf,
            force_buf,
            rng_buf,
            volume,
            dt,
            "kick",
        )
    }

    /// U ← exp(dt × π) × U  then reunitarize
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn link_update(
        &self,
        links_buf: &wgpu::Buffer,
        momenta_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        dt: f64,
    ) -> Result<()> {
        self.dispatch(
            &self.update_pipeline,
            links_buf,
            momenta_buf,
            force_buf,
            rng_buf,
            volume,
            dt,
            "update",
        )
    }

    /// Generate random su(3) algebra momenta.
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn generate_momenta(
        &self,
        links_buf: &wgpu::Buffer,
        momenta_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
    ) -> Result<()> {
        self.dispatch(
            &self.gen_pipeline,
            links_buf,
            momenta_buf,
            force_buf,
            rng_buf,
            volume,
            0.0,
            "gen",
        )
    }

    #[expect(clippy::too_many_arguments, reason = "API")]
    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        links_buf: &wgpu::Buffer,
        momenta_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        dt: f64,
        label: &str,
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

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("GpuHmcLeapfrog:{label}:bg")),
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
                        resource: momenta_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: force_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: rng_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("GpuHmcLeapfrog:{label}:enc")),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("GpuHmcLeapfrog:{label}:pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(self.n_links.div_ceil(WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }

    /// Number of gauge links (volume × 4).
    #[must_use]
    pub fn n_links(&self) -> u32 {
        self.n_links
    }
}

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
    fn test_leapfrog_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuHmcLeapfrog::new(device, 16).unwrap();
        assert_eq!(op.n_links(), 64);
    }
}
