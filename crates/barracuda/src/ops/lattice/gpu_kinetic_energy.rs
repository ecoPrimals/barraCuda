// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU per-link kinetic energy from HMC momenta.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::su3::{su3_df64_preamble, su3_preamble};

const WG: u32 = 64;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/kinetic_energy_f64.wgsl");
const SHADER_DF64: &str = include_str!("../../shaders/lattice/kinetic_energy_df64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KineticParams {
    n_links: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU kinetic energy operator: -0.5 × Re Tr(π²) per link.
pub struct GpuKineticEnergy {
    device: Arc<WgpuDevice>,
    n_links: u32,
    shader_src: String,
    params: wgpu::Buffer,
}

impl GpuKineticEnergy {
    /// Create kinetic energy calculator for given lattice volume.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let n_links = volume * 4;

        let profile = GpuDriverProfile::from_device(&device);
        let strategy = profile.fp64_strategy();
        let shader_src = match strategy {
            Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                format!("{}{}", su3_preamble(), SHADER_BODY)
            }
            Fp64Strategy::Hybrid => format!("{}{}", su3_df64_preamble(), SHADER_DF64),
        };
        tracing::info!(
            ?strategy,
            "GpuKineticEnergy: compiled with {:?} FP64 strategy",
            strategy
        );

        let params_data = KineticParams {
            n_links,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuKineticEnergy:params"),
            size: std::mem::size_of::<KineticParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            n_links,
            shader_src,
            params,
        })
    }

    /// Compute per-link kinetic energy.
    ///
    /// * `momenta_buf` — `[V × 4 × 18]` f64 (conjugate momenta)
    /// * `energy_buf`  — `[V × 4]` f64 (per-link kinetic energy)
    pub fn compute(&self, momenta_buf: &wgpu::Buffer, energy_buf: &wgpu::Buffer) -> Result<()> {
        ComputeDispatch::new(self.device.as_ref(), "GpuKineticEnergy")
            .shader(&self.shader_src, "kinetic_energy_kernel")
            .f64()
            .uniform(0, &self.params)
            .storage_read(1, momenta_buf)
            .storage_rw(2, energy_buf)
            .dispatch(self.n_links.div_ceil(WG), 1, 1)
            .submit();
        Ok(())
    }

    /// Number of gauge links (volume × 4).
    pub fn n_links(&self) -> u32 {
        self.n_links
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kinetic_energy_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuKineticEnergy::new(device, 16).unwrap();
        assert_eq!(op.n_links(), 64);
    }
}
