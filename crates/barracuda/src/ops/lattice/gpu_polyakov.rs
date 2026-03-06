// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Polyakov loop (temporal Wilson line) computation.

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::su3::su3_preamble;

const WG: u32 = 64;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/polyakov_loop_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PolyParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    spatial_vol: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU Polyakov loop operator.
pub struct GpuPolyakovLoop {
    device: Arc<WgpuDevice>,
    spatial_vol: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

impl GpuPolyakovLoop {
    /// Create Polyakov loop calculator for given lattice dimensions.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>, nt: u32, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let volume = nt * nx * ny * nz;
        let spatial_vol = nx * ny * nz;
        let src = format!("{}{}", su3_preamble(), SHADER_BODY);
        let module = device.compile_shader_f64(&src, Some("polyakov_loop"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuPolyakovLoop:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),  // links
                    storage_bgl(2, false), // poly output
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuPolyakovLoop:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuPolyakovLoop:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("polyakov_loop_kernel"),
                compilation_options: Default::default(),
                cache: None,
            });

        let params_data = PolyParams {
            nt,
            nx,
            ny,
            nz,
            volume,
            spatial_vol,
            _pad0: 0,
            _pad1: 0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuPolyakovLoop:params"),
            size: std::mem::size_of::<PolyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            spatial_vol,
            pipeline,
            bgl,
            params,
        })
    }

    /// Compute Polyakov loop for all spatial sites.
    /// * `links_buf` — `[V × 4 × 18]` f64
    /// * `poly_buf`  — `[spatial_vol × 2]` f64 (Re, Im per spatial site)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(&self, links_buf: &wgpu::Buffer, poly_buf: &wgpu::Buffer) -> Result<()> {
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GpuPolyakovLoop:bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: links_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: poly_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuPolyakovLoop:enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuPolyakovLoop:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(self.spatial_vol.div_ceil(WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }

    /// Spatial volume (nx × ny × nz).
    #[must_use]
    pub fn spatial_vol(&self) -> u32 {
        self.spatial_vol
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
    fn test_polyakov_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuPolyakovLoop::new(device, 4, 2, 2, 2).unwrap();
        assert_eq!(op.spatial_vol(), 8);
    }
}
