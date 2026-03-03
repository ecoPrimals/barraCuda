// SPDX-License-Identifier: AGPL-3.0-or-later
//! 1D gradient computation: df/dx

use super::super::fd_common::{create_staging_buffer, FdPipelineBuilder, FD_WORKGROUP_SIZE};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// 1D gradient computation
pub struct Gradient1D {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    n: usize,
    dx: f64,
}

impl Gradient1D {
    /// Create a new 1D gradient operator
    pub fn new(device: Arc<WgpuDevice>, n: usize, dx: f64) -> Result<Self> {
        if n == 0 {
            return Err(BarracudaError::invalid_op(
                "Gradient1D",
                "n must be > 0 (zero-length buffers are invalid for GPU compute)",
            ));
        }
        let (pipeline, bind_group_layout) = FdPipelineBuilder::new(device.device(), "gradient_1d")
            .with_uniform(0)
            .with_input(1)
            .with_output(2)
            .build("gradient_1d")?;

        Ok(Self {
            device,
            pipeline,
            bind_group_layout,
            n,
            dx,
        })
    }

    /// Compute gradient df/dx
    pub async fn compute(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Input size mismatch: expected {}, got {}",
                    self.n,
                    input.len()
                ),
            });
        }

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n: u32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
            dx: f64,
        }

        let params = Params {
            n: self.n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            dx: self.dx,
        };

        let params_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("grad1d_params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let input_buffer =
            self.device
                .device()
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("grad1d_input"),
                    contents: bytemuck::cast_slice(input),
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let output_buffer = self.device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("grad1d_output"),
            size: (self.n * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group = self
            .device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("grad1d_bg"),
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
                        resource: output_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("grad1d"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(self.n.div_ceil(FD_WORKGROUP_SIZE as usize) as u32, 1, 1);
        }

        let buffer_size = (self.n * std::mem::size_of::<f64>()) as u64;
        let staging = create_staging_buffer(self.device.device(), buffer_size, "grad1d_staging");
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging, 0, buffer_size);
        self.device.queue().submit(Some(encoder.finish()));

        self.device.map_staging_buffer::<f64>(&staging, self.n)
    }
}
