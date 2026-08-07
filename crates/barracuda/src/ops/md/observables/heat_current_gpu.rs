// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU heat current for Green-Kubo thermal conductivity (f64).
//!
//! Computes the microscopic heat current `J_q` per particle from positions,
//! velocities, and Yukawa interaction parameters. Output is per-particle
//! [`J_x`, `J_y`, `J_z`] f64 vectors; host sums to get total `J_q(t)`.
//!
//! Absorbed from hotSpring CPU `compute_heat_current()` → GPU shader.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../../shaders/md/observables/heat_current_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HeatParams {
    n: u32,
    _pad0: u32,
    box_side: f64,
    kappa: f64,
    mass: f64,
}

/// Per-particle heat current GPU kernel (Yukawa interaction, f64).
pub struct HeatCurrentGpu {
    device: Arc<WgpuDevice>,
}

impl HeatCurrentGpu {
    /// Creates a new heat current GPU kernel for the given WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Dispatch heat current computation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation or GPU dispatch fails (e.g. device lost).
    ///
    /// * `pos_buf` — `[N × 3]` f64 positions
    /// * `vel_buf` — `[N × 3]` f64 velocities
    /// * `jq_buf`  — `[N × 3]` f64 output (per-particle `J_q`)
    pub fn dispatch(
        &self,
        pos_buf: &wgpu::Buffer,
        vel_buf: &wgpu::Buffer,
        jq_buf: &wgpu::Buffer,
        n: u32,
        box_side: f64,
        kappa: f64,
        mass: f64,
    ) -> Result<()> {
        let params_data = HeatParams {
            n,
            _pad0: 0,
            box_side,
            kappa,
            mass,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HeatCurrent:params"),
            size: std::mem::size_of::<HeatParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        ComputeDispatch::new(&self.device, "HeatCurrent")
            .shader(SHADER, "heat_current")
            .f64()
            .uniform(0, &params)
            .storage_read(1, pos_buf)
            .storage_read(2, vel_buf)
            .storage_rw(3, jq_buf)
            .dispatch(n.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_current_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let hc = HeatCurrentGpu::new(device).unwrap();
        assert!(std::mem::size_of_val(&hc) > 0);
    }
}
