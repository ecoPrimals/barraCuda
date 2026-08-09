// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU pseudofermion operations: heatbath noise and fermion force.
//!
//! The heatbath generates Gaussian noise η; the actual φ = D†η is performed
//! by dispatching the staggered Dirac operator from `dirac.rs`.
//!
//! The fermion force computes `dS_F/dU` from CG solution fields.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

use super::complex_f64::WGSL_COMPLEX64;
use super::lcg::WGSL_LCG_F64;
use super::su3::su3_preamble;
const HEATBATH_SHADER: &str = include_str!("../../shaders/lattice/pseudofermion_heatbath_f64.wgsl");
const FORCE_SHADER: &str = include_str!("../../shaders/lattice/pseudofermion_force_f64.wgsl");

// ── Heatbath ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HeatbathParams {
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU pseudofermion heatbath — generates Gaussian noise fermion field.
pub struct GpuPseudofermionHeatbath {
    device: Arc<WgpuDevice>,
    volume: u32,
    shader_src: String,
    params: wgpu::Buffer,
}

impl GpuPseudofermionHeatbath {
    /// Create a heatbath operator for the given lattice volume.
    /// # Errors
    /// Returns [`Err`] if shader compilation fails, buffer allocation fails, or the device is lost.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let shader_src = format!("{WGSL_COMPLEX64}\n{WGSL_LCG_F64}\n{HEATBATH_SHADER}");

        let params_data = HeatbathParams {
            volume,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuPfHeatbath:params"),
            size: std::mem::size_of::<HeatbathParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            volume,
            shader_src,
            params,
        })
    }

    /// Generate Gaussian noise into `eta_buf`.
    /// * `eta_buf`     — `[V × 6]` f64 (3 colors × 2)
    /// * `rng_buf`     — `[V]` u64 (per-site RNG state)
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn generate(&self, eta_buf: &wgpu::Buffer, rng_buf: &wgpu::Buffer) -> Result<()> {
        ComputeDispatch::new(&self.device, "GpuPfHeatbath")
            .shader(&self.shader_src, "heatbath_noise")
            .f64()
            .uniform(0, &self.params)
            .storage_rw(1, eta_buf)
            .storage_rw(2, rng_buf)
            .dispatch(self.volume.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }

    /// Return the lattice volume.
    #[must_use]
    pub fn volume(&self) -> u32 {
        self.volume
    }
}

// ── Pseudofermion Force ─────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PFForceParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU pseudofermion force: `dS_F/dU` from CG solution fields.
pub struct GpuPseudofermionForce {
    device: Arc<WgpuDevice>,
    volume: u32,
    shader_src: String,
    params: wgpu::Buffer,
}

impl GpuPseudofermionForce {
    /// Create a pseudofermion force operator for the given lattice dimensions.
    /// # Errors
    /// Returns [`Err`] if shader compilation fails, buffer allocation fails, or the device is lost.
    pub fn new(device: Arc<WgpuDevice>, nt: u32, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let volume = nt * nx * ny * nz;
        let shader_src = format!("{}{}", su3_preamble(), FORCE_SHADER);

        let params_data = PFForceParams {
            nt,
            nx,
            ny,
            nz,
            volume,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuPfForce:params"),
            size: std::mem::size_of::<PFForceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            volume,
            shader_src,
            params,
        })
    }

    /// Compute pseudofermion force for all links.
    /// * `links_buf`   — `[V × 4 × 18]` f64
    /// * `x_field_buf` — `[V × 6]` f64 (CG solution)
    /// * `y_field_buf` — `[V × 6]` f64 (D·X)
    /// * `force_buf`   — `[V × 4 × 18]` f64 (output)
    /// # Errors
    /// Returns [`Err`] if buffer sizes are invalid for the volume, command submission fails, or the device is lost.
    pub fn compute(
        &self,
        links_buf: &wgpu::Buffer,
        x_field_buf: &wgpu::Buffer,
        y_field_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
    ) -> Result<()> {
        ComputeDispatch::new(&self.device, "GpuPfForce")
            .shader(&self.shader_src, "pseudofermion_force_kernel")
            .f64()
            .uniform(0, &self.params)
            .storage_read(1, links_buf)
            .storage_read(2, x_field_buf)
            .storage_read(3, y_field_buf)
            .storage_rw(4, force_buf)
            .dispatch(self.volume.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }

    /// Lattice volume.
    #[must_use]
    pub fn volume(&self) -> u32 {
        self.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heatbath_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuPseudofermionHeatbath::new(device, 16).unwrap();
        assert_eq!(op.volume(), 16);
    }

    #[test]
    fn test_pf_force_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuPseudofermionForce::new(device, 2, 2, 2, 2).unwrap();
        assert_eq!(op.volume(), 16);
    }
}
