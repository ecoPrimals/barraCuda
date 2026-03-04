// SPDX-License-Identifier: AGPL-3.0-or-later
//! 2D Laplacian computation: ∇²f = ∂²f/∂x² + ∂²f/∂y²

use super::super::fd_common::FdPipelineBuilder;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// 2D Laplacian computation: ∇²f = ∂²f/∂x² + ∂²f/∂y²
pub struct Laplacian2D {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
}

impl Laplacian2D {
    /// Create a new 2D Laplacian operator
    pub fn new(device: Arc<WgpuDevice>, nx: usize, ny: usize, dx: f64, dy: f64) -> Result<Self> {
        let (pipeline, bind_group_layout) = FdPipelineBuilder::new(device.device(), "laplacian_2d")
            .with_uniform(0)
            .with_input(1)
            .with_output(2)
            .with_output(3)
            .with_output(4)
            .with_output(5)
            .build("laplacian_2d")?;

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            nx,
            ny,
            dx,
            dy,
        })
    }

    /// Grid dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.nx, self.ny)
    }

    /// Compute 2D Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²
    pub async fn compute(&self, input: &[f64]) -> Result<Vec<f64>> {
        let total = self.nx * self.ny;
        if input.len() != total {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input size mismatch: expected {} ({}×{}), got {}",
                    total,
                    self.nx,
                    self.ny,
                    input.len()
                ),
            });
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            nx: u32,
            ny: u32,
            _pad0: u32,
            _pad1: u32,
            dx: f64,
            dy: f64,
        }

        let params = Params {
            nx: self.nx as u32,
            ny: self.ny as u32,
            _pad0: 0,
            _pad1: 0,
            dx: self.dx,
            dy: self.dy,
        };

        let params_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("lap2d_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let input_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("lap2d_input"),
                    contents: bytemuck::cast_slice(input),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let buffer_size = (total * std::mem::size_of::<f64>()) as u64;

        let dummy_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("lap2d_dummy"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let laplacian_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("lap2d_output"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("lap2d_bg"),
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
                        resource: dummy_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: laplacian_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lap2d"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(self.nx.div_ceil(16) as u32, self.ny.div_ceil(16) as u32, 1);
        }

        let staging = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("lap2d_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&laplacian_buffer, 0, &staging, 0, buffer_size);
        self.device.submit_commands(Some(encoder.finish()));

        self.device.map_staging_buffer::<f64>(&staging, total)
    }
}
