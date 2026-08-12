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

/// Per-link workgroup size — must match @workgroup_size in leapfrog shaders.
/// 128 keeps 32⁴ (65536 links/WG at WG64) under the 65535 dispatch limit.
const WG_LINK: u32 = 128;

use super::su3::su3_df64_preamble;
use super::su3_extended::su3_extended_preamble;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/hmc_leapfrog_f64.wgsl");
const SHADER_MOMENTUM_DF64: &str =
    include_str!("../../shaders/lattice/su3_momentum_update_df64.wgsl");
const SHADER_LINK_DF64: &str = include_str!("../../shaders/lattice/su3_link_update_df64.wgsl");

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

/// GPU HMC leapfrog integrator with strategy-aware dispatch.
///
/// In `Concurrent` mode, momentum kick and link update use DF64 shaders
/// (throughput-bound work on FP32 cores), while momentum generation uses
/// native f64 (precision-critical RNG).
pub struct GpuHmcLeapfrog {
    device: Arc<WgpuDevice>,
    n_links: u32,
    native_shader_src: String,
    df64_momentum_src: Option<String>,
    df64_link_src: Option<String>,
    strategy: Fp64Strategy,
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
        })
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
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn generate_momenta(&self, buffers: &LeapfrogBuffers<'_>, volume: u32) -> Result<()> {
        self.dispatch_native(buffers, volume, 0.0, "generate_momenta", "gen")
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
        volume: u32,
        dt: f64,
        shader_src: &str,
        entry_point: &str,
        label: &str,
    ) -> Result<()> {
        self.dispatch_shader(shader_src, buffers, volume, dt, entry_point, label, false)
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
