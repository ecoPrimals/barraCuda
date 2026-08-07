// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Polyakov loop (temporal Wilson line) computation.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::Result;
use std::sync::Arc;

use super::su3::su3_preamble;
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
    shader_src: String,
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
        let shader_src = format!("{}{}", su3_preamble(), SHADER_BODY);

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
            shader_src,
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
        ComputeDispatch::new(&self.device, "GpuPolyakovLoop")
            .shader(&self.shader_src, "polyakov_loop_kernel")
            .f64()
            .uniform(0, &self.params)
            .storage_read(1, links_buf)
            .storage_rw(2, poly_buf)
            .dispatch(self.spatial_vol.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }

    /// Spatial volume (nx × ny × nz).
    #[must_use]
    pub fn spatial_vol(&self) -> u32 {
        self.spatial_vol
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
