// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cylindrical (ρ, z) gradient and Laplacian for axially symmetric problems
//!
//! Used for nuclear physics (deformed nuclei), fluid dynamics, etc.

use super::super::fd_common::{FdPipelineBuilder, FD_WORKGROUP_SIZE};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Cylindrical (ρ, z) gradient: ∂f/∂ρ and ∂f/∂z
pub struct CylindricalGradient {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    n_rho: usize,
    n_z: usize,
    d_rho: f64,
    d_z: f64,
    z_min: f64,
}

impl CylindricalGradient {
    /// Create a new cylindrical gradient operator
    pub fn new(
        device: Arc<WgpuDevice>,
        n_rho: usize,
        n_z: usize,
        d_rho: f64,
        d_z: f64,
        z_min: f64,
    ) -> Result<Self> {
        let (pipeline, bind_group_layout) = FdPipelineBuilder::new(device.device(), "cyl_grad")
            .with_uniform(0)
            .with_input(1)
            .with_output(2)
            .with_output(3)
            .build("gradient_cylindrical")?;

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            n_rho,
            n_z,
            d_rho,
            d_z,
            z_min,
        })
    }

    /// Grid dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rho, self.n_z)
    }

    /// Compute cylindrical gradient (∂f/∂ρ, ∂f/∂z)
    pub async fn compute(&self, input: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let total = self.n_rho * self.n_z;
        if input.len() != total {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input size mismatch: expected {} ({}×{}), got {}",
                    total,
                    self.n_rho,
                    self.n_z,
                    input.len()
                ),
            });
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct CylParams {
            n_rho: u32,
            n_z: u32,
            _pad0: u32,
            _pad1: u32,
            d_rho: f64,
            d_z: f64,
            z_min: f64,
        }

        let params = CylParams {
            n_rho: self.n_rho as u32,
            n_z: self.n_z as u32,
            _pad0: 0,
            _pad1: 0,
            d_rho: self.d_rho,
            d_z: self.d_z,
            z_min: self.z_min,
        };

        let params_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cyl_grad_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let input_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cyl_grad_input"),
                    contents: bytemuck::cast_slice(input),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let buffer_size = (total * std::mem::size_of::<f64>()) as u64;

        let grad_rho_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cyl_grad_rho"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let grad_z_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cyl_grad_z"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cyl_grad_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: grad_rho_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: grad_z_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cyl_grad"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total.div_ceil(FD_WORKGROUP_SIZE as usize) as u32, 1, 1);
        }

        let staging_rho = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_rho"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging_z = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_z"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&grad_rho_buffer, 0, &staging_rho, 0, buffer_size);
        encoder.copy_buffer_to_buffer(&grad_z_buffer, 0, &staging_z, 0, buffer_size);
        self.device.submit_commands(Some(encoder.finish()));

        let grad_rho = self.device.map_staging_buffer::<f64>(&staging_rho, total)?;
        let grad_z = self.device.map_staging_buffer::<f64>(&staging_z, total)?;

        Ok((grad_rho, grad_z))
    }
}

/// Cylindrical Laplacian: ∇²f = ∂²f/∂ρ² + (1/ρ)∂f/∂ρ + ∂²f/∂z²
pub struct CylindricalLaplacian {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    n_rho: usize,
    n_z: usize,
    d_rho: f64,
    d_z: f64,
    z_min: f64,
}

impl CylindricalLaplacian {
    /// Create a new cylindrical Laplacian operator
    pub fn new(
        device: Arc<WgpuDevice>,
        n_rho: usize,
        n_z: usize,
        d_rho: f64,
        d_z: f64,
        z_min: f64,
    ) -> Result<Self> {
        let (pipeline, bind_group_layout) = FdPipelineBuilder::new(device.device(), "cyl_lap")
            .with_uniform(0)
            .with_input(1)
            .with_output(2)
            .with_output(3)
            .with_output(4)
            .build("laplacian_cylindrical")?;

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            n_rho,
            n_z,
            d_rho,
            d_z,
            z_min,
        })
    }

    /// Grid dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.n_rho, self.n_z)
    }

    /// Compute cylindrical Laplacian: ∇²f = ∂²f/∂ρ² + (1/ρ)∂f/∂ρ + ∂²f/∂z²
    pub async fn compute(&self, input: &[f64]) -> Result<Vec<f64>> {
        let total = self.n_rho * self.n_z;
        if input.len() != total {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input size mismatch: expected {} ({}×{}), got {}",
                    total,
                    self.n_rho,
                    self.n_z,
                    input.len()
                ),
            });
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct CylParams {
            n_rho: u32,
            n_z: u32,
            _pad0: u32,
            _pad1: u32,
            d_rho: f64,
            d_z: f64,
            z_min: f64,
        }

        let params = CylParams {
            n_rho: self.n_rho as u32,
            n_z: self.n_z as u32,
            _pad0: 0,
            _pad1: 0,
            d_rho: self.d_rho,
            d_z: self.d_z,
            z_min: self.z_min,
        };

        let params_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cyl_lap_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let input_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("cyl_lap_input"),
                    contents: bytemuck::cast_slice(input),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let buffer_size = (total * std::mem::size_of::<f64>()) as u64;

        let dummy_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cyl_lap_dummy"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let laplacian_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cyl_lap_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("cyl_lap_bg"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: input_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dummy_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dummy_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: laplacian_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cyl_lap"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(total.div_ceil(FD_WORKGROUP_SIZE as usize) as u32, 1, 1);
        }

        let staging = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("cyl_lap_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&laplacian_buffer, 0, &staging, 0, buffer_size);
        self.device.submit_commands(Some(encoder.finish()));

        self.device.map_staging_buffer::<f64>(&staging, total)
    }
}
