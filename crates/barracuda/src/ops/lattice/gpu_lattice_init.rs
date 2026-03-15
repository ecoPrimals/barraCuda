// SPDX-License-Identifier: AGPL-3.0-only
//! GPU lattice initialization: cold start and hot start.
//!
//! Replaces CPU-only `wilson.rs` `cold_start/hot_start` with GPU shaders.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::Result;
use std::sync::Arc;

use super::su3_extended::su3_extended_preamble;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/lattice_init_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InitParams {
    volume: u32,
    n_links: u32,
    _pad0: u32,
    _pad1: u32,
    epsilon: f64,
    _padf: f64,
}

/// GPU lattice initializer — cold start (identity) or hot start (random near identity).
pub struct GpuLatticeInit {
    device: Arc<WgpuDevice>,
    n_links: u32,
    cold_pipeline: wgpu::ComputePipeline,
    hot_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl GpuLatticeInit {
    /// Create lattice initializer for given volume.
    /// # Errors
    /// Returns [`Err`] if shader compilation or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let n_links = volume * 4;
        let src = format!("{}{}", su3_extended_preamble(), SHADER_BODY);
        let module = device.compile_shader_f64(&src, Some("lattice_init"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuLatticeInit:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, false), // links (write)
                    storage_bgl(2, false), // rng_state (read_write)
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuLatticeInit:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let cold_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GpuLatticeInit:cold"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("cold_start"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let hot_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GpuLatticeInit:hot"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("hot_start"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Ok(Self {
            device,
            n_links,
            cold_pipeline,
            hot_pipeline,
            bgl,
        })
    }

    /// Initialize all links to SU(3) identity.
    /// # Errors
    /// Returns [`Err`] if GPU dispatch fails or the device is lost.
    pub fn cold_start(
        &self,
        links_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
    ) -> Result<()> {
        self.dispatch(
            &self.cold_pipeline,
            links_buf,
            rng_buf,
            volume,
            0.0,
            "cold_start",
        )
    }

    /// Initialize links with random SU(3) near identity.
    /// # Errors
    /// Returns [`Err`] if GPU dispatch fails or the device is lost.
    pub fn hot_start(
        &self,
        links_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        epsilon: f64,
    ) -> Result<()> {
        self.dispatch(
            &self.hot_pipeline,
            links_buf,
            rng_buf,
            volume,
            epsilon,
            "hot_start",
        )
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        links_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        epsilon: f64,
        label: &str,
    ) -> Result<()> {
        let params_data = InitParams {
            volume,
            n_links: self.n_links,
            _pad0: 0,
            _pad1: 0,
            epsilon,
            _padf: 0.0,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("GpuLatticeInit:{label}:params")),
            size: std::mem::size_of::<InitParams>() as u64,
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
                label: Some(&format!("GpuLatticeInit:{label}:bg")),
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
                        resource: rng_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some(&format!("GpuLatticeInit:{label}:enc")),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("GpuLatticeInit:{label}:pass")),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(self.n_links.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1);
        }
        self.device.submit_commands(Some(enc.finish()));
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
    fn test_init_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let init = GpuLatticeInit::new(device, 16).unwrap();
        assert_eq!(init.n_links(), 64);
    }

    #[test]
    fn test_cold_start_identity_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };

        let volume = 16u32;
        let n_links = volume * 4;
        let init = GpuLatticeInit::new(device.clone(), volume).unwrap();

        let links_bytes = (n_links as usize) * 18 * std::mem::size_of::<f64>();
        let links_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:links"),
            size: links_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rng_bytes = (n_links as usize) * std::mem::size_of::<u32>();
        let rng_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:rng"),
            size: rng_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        init.cold_start(&links_buf, &rng_buf, volume).unwrap();

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:staging"),
            size: links_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&links_buf, 0, &staging, 0, links_bytes as u64);
        device.submit_commands(Some(enc.finish()));

        let n_f64 = links_bytes / std::mem::size_of::<f64>();
        let data: Vec<f64> = device.map_staging_buffer(&staging, n_f64).unwrap();

        for link in 0..n_links as usize {
            for i in 0..9 {
                let re = data[link * 18 + i * 2];
                let im = data[link * 18 + i * 2 + 1];
                let (exp_re, exp_im) = if i == 0 || i == 4 || i == 8 {
                    (1.0, 0.0)
                } else {
                    (0.0, 0.0)
                };
                assert!(
                    (re - exp_re).abs() < 1e-10 && (im - exp_im).abs() < 1e-10,
                    "link {link} elem {i}: ({re}, {im}) expected ({exp_re}, {exp_im})"
                );
            }
        }
    }
}
