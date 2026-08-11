// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU HMC leapfrog integration: momentum kick, link update, momentum generation.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

/// Per-link workgroup size — must match @workgroup_size in hmc_leapfrog_f64.wgsl.
/// 128 keeps 32⁴ (65536 links/WG at WG64) under the 65535 dispatch limit.
const WG_LINK: u32 = 128;

use super::su3_extended::su3_extended_preamble;
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

/// GPU HMC leapfrog integrator with three dispatch modes.
pub struct GpuHmcLeapfrog {
    device: Arc<WgpuDevice>,
    n_links: u32,
    shader_src: String,
}

impl GpuHmcLeapfrog {
    /// Create HMC leapfrog integrator for given lattice volume.
    /// # Errors
    /// Returns [`Err`] if shader compilation fails, pipeline creation fails, or the device is lost.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let n_links = volume * 4;
        let shader_src = format!("{}{}", su3_extended_preamble(), SHADER_BODY);

        Ok(Self {
            device,
            n_links,
            shader_src,
        })
    }

    /// π ← π + dt × force
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn momentum_kick(&self, buffers: &LeapfrogBuffers<'_>, volume: u32, dt: f64) -> Result<()> {
        self.dispatch(buffers, volume, dt, "momentum_kick", "kick")
    }

    /// U ← exp(dt × π) × U  then reunitarize
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn link_update(&self, buffers: &LeapfrogBuffers<'_>, volume: u32, dt: f64) -> Result<()> {
        self.dispatch(buffers, volume, dt, "link_update", "update")
    }

    /// Generate random su(3) algebra momenta.
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn generate_momenta(&self, buffers: &LeapfrogBuffers<'_>, volume: u32) -> Result<()> {
        self.dispatch(buffers, volume, 0.0, "generate_momenta", "gen")
    }

    fn dispatch(
        &self,
        buffers: &LeapfrogBuffers<'_>,
        volume: u32,
        dt: f64,
        entry_point: &str,
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

        ComputeDispatch::new(&self.device, &format!("GpuHmcLeapfrog:{label}"))
            .shader(&self.shader_src, entry_point)
            .f64()
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
