// SPDX-License-Identifier: AGPL-3.0-or-later
//! VELOCITY-VERLET F64 — Symplectic integrator — f64 WGSL shader dispatch
//!
//! All math originates as `velocity_verlet_f64.wgsl`.
//! Three entry points: `main` (full step), `velocity_half_step`, `position_update`.
//!
//! Applications:
//! - Molecular dynamics
//! - N-body simulations
//! - Long-time energy conservation

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("velocity_verlet_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VvParams {
    n_particles: u32,
    _pad0: u32,
    dt: f64,
}

/// GPU-accelerated Velocity-Verlet integrator (f64).
///
/// Algorithm:
/// 1. x(t+Δt) = x(t) + v(t)Δt + ½a(t)Δt²
/// 2. v(t+Δt) = v(t) + ½[a(t) + a(t+Δt)]Δt
pub struct VelocityVerletF64 {
    device: Arc<WgpuDevice>,
}

impl VelocityVerletF64 {
    /// Create f64 Velocity-Verlet integrator.
    /// # Errors
    /// Returns [`Err`] if device initialization fails (currently always succeeds).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Full Velocity-Verlet step on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, staging buffer mapping fails, or the device is lost.
    pub fn step(
        &self,
        positions: &[f64],
        velocities: &[f64],
        forces_old: &[f64],
        forces_new: &[f64],
        masses: &[f64],
        dt: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = positions.len() / 3;
        let n3 = n * 3;

        let d = &self.device.device;

        let buf = |label, data: &[f64]| {
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        let pos_buf = buf("VV:pos", positions);
        let vel_buf = buf("VV:vel", velocities);
        let fo_buf = buf("VV:fo", forces_old);
        let fn_buf = buf("VV:fn", forces_new);
        let mass_buf = buf("VV:m", masses);

        let out_size = (n3 * 8) as u64;
        let make_out = |label| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: out_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let pos_out = make_out("VV:pos_out");
        let vel_out = make_out("VV:vel_out");

        let params = VvParams {
            n_particles: n as u32,
            _pad0: 0,
            dt,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wg_count = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "VV:step")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &pos_buf)
            .storage_read(1, &vel_buf)
            .storage_read(2, &fo_buf)
            .storage_read(3, &fn_buf)
            .storage_read(4, &mass_buf)
            .storage_rw(5, &pos_out)
            .storage_rw(6, &vel_out)
            .uniform(7, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        let rb_pos = readback_buf(d, out_size);
        let rb_vel = readback_buf(d, out_size);
        let mut enc = self.device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&pos_out, 0, &rb_pos, 0, out_size);
        enc.copy_buffer_to_buffer(&vel_out, 0, &rb_vel, 0, out_size);
        self.device.submit_and_poll(Some(enc.finish()));

        let new_pos = self.device.map_staging_buffer::<f64>(&rb_pos, n3)?;
        let new_vel = self.device.map_staging_buffer::<f64>(&rb_vel, n3)?;

        Ok((new_pos, new_vel))
    }

    /// Half-step velocity update on GPU (first half of leapfrog).
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, staging buffer mapping fails, or the device is lost.
    pub fn velocity_half_step(
        &self,
        velocities: &[f64],
        forces: &[f64],
        masses: &[f64],
        dt: f64,
    ) -> Result<Vec<f64>> {
        let n = velocities.len() / 3;
        let n3 = n * 3;
        let d = &self.device.device;

        let vel_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:hv_vel"),
            contents: bytemuck::cast_slice(velocities),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let forces_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:hv_f"),
            contents: bytemuck::cast_slice(forces),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mass_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:hv_m"),
            contents: bytemuck::cast_slice(masses),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_size = (n3 * 8) as u64;
        let ph_ro = self.device.placeholder_buffer();
        let ph_rw = self.device.placeholder_buffer_rw();
        let vel_out = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VV:hv_out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = VvParams {
            n_particles: n as u32,
            _pad0: 0,
            dt,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:hv_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wg_count = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "VV:half_vel")
            .shader(SHADER, "velocity_half_step")
            .f64()
            .storage_read(0, ph_ro)
            .storage_read(1, &vel_buf)
            .storage_read(2, &forces_buf)
            .storage_read(3, ph_ro)
            .storage_read(4, &mass_buf)
            .storage_rw(5, ph_rw)
            .storage_rw(6, &vel_out)
            .uniform(7, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        let rb = readback_buf(d, out_size);
        let mut enc = self.device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&vel_out, 0, &rb, 0, out_size);
        self.device.submit_and_poll(Some(enc.finish()));

        self.device.map_staging_buffer::<f64>(&rb, n3)
    }

    /// Position update on GPU using velocities.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, staging buffer mapping fails, or the device is lost.
    pub fn position_update(
        &self,
        positions: &[f64],
        velocities: &[f64],
        dt: f64,
    ) -> Result<Vec<f64>> {
        let n = positions.len() / 3;
        let n3 = n * 3;
        let d = &self.device.device;

        let pos_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:pu_pos"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let vel_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:pu_vel"),
            contents: bytemuck::cast_slice(velocities),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_size = (n3 * 8) as u64;
        let ph_ro = self.device.placeholder_buffer();
        let ph_rw = self.device.placeholder_buffer_rw();
        let pos_out = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VV:pu_out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = VvParams {
            n_particles: n as u32,
            _pad0: 0,
            dt,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VV:pu_p"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wg_count = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "VV:pos_update")
            .shader(SHADER, "position_update")
            .f64()
            .storage_read(0, &pos_buf)
            .storage_read(1, &vel_buf)
            .storage_read(2, ph_ro)
            .storage_read(3, ph_ro)
            .storage_read(4, ph_ro)
            .storage_rw(5, &pos_out)
            .storage_rw(6, ph_rw)
            .uniform(7, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        let rb = readback_buf(d, out_size);
        let mut enc = self.device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&pos_out, 0, &rb, 0, out_size);
        self.device.submit_and_poll(Some(enc.finish()));

        self.device.map_staging_buffer::<f64>(&rb, n3)
    }
}

fn readback_buf(d: &wgpu::Device, size: u64) -> wgpu::Buffer {
    d.create_buffer(&wgpu::BufferDescriptor {
        label: Some("VV:readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_free_particle() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let vv = VelocityVerletF64::new(device)?;

        // Particle moving with constant velocity (no force)
        let pos = vec![0.0, 0.0, 0.0];
        let vel = vec![1.0, 2.0, 3.0];
        let forces = vec![0.0, 0.0, 0.0];
        let masses = vec![1.0];
        let dt = 0.1;

        let (new_pos, new_vel) = vv.step(&pos, &vel, &forces, &forces, &masses, dt)?;

        // Position: x = x₀ + v*dt
        assert!((new_pos[0] - 0.1).abs() < 1e-10);
        assert!((new_pos[1] - 0.2).abs() < 1e-10);
        assert!((new_pos[2] - 0.3).abs() < 1e-10);

        // Velocity unchanged (no acceleration)
        assert!((new_vel[0] - 1.0).abs() < 1e-10);
        assert!((new_vel[1] - 2.0).abs() < 1e-10);
        assert!((new_vel[2] - 3.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_constant_acceleration() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let vv = VelocityVerletF64::new(device)?;

        // Particle under constant force
        let pos = vec![0.0, 0.0, 0.0];
        let vel = vec![0.0, 0.0, 0.0];
        let forces = vec![1.0, 0.0, 0.0]; // F = 1 in x
        let masses = vec![1.0];
        let dt = 0.1;

        let (new_pos, new_vel) = vv.step(&pos, &vel, &forces, &forces, &masses, dt)?;

        // a = F/m = 1
        // x = ½at² = 0.005
        // v = at = 0.1
        assert!((new_pos[0] - 0.005).abs() < 1e-10);
        assert!((new_vel[0] - 0.1).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_symplectic_energy_conservation() {
        let Some(device) = create_test_device() else {
            return;
        };
        let result = (|| -> Result<()> {
            let vv = VelocityVerletF64::new(device)?;

            let mut pos = vec![1.0, 0.0, 0.0];
            let mut vel = vec![0.0, 0.0, 0.0];
            let masses = vec![1.0];
            let dt = 0.01;

            let initial_energy = 0.5 * pos[0] * pos[0] + 0.5 * vel[0] * vel[0];

            for _ in 0..1000 {
                let forces_old = vec![-pos[0], 0.0, 0.0];
                let half_vel = vv.velocity_half_step(&vel, &forces_old, &masses, dt)?;
                pos = vv.position_update(&pos, &half_vel, dt)?;
                let forces_new = vec![-pos[0], 0.0, 0.0];
                vel = vv.velocity_half_step(&half_vel, &forces_new, &masses, dt)?;
            }

            let final_energy = 0.5 * pos[0] * pos[0] + 0.5 * vel[0] * vel[0];
            let rel_err = (final_energy - initial_energy).abs() / initial_energy;

            assert!(
                rel_err < 1e-4,
                "Energy drift {} too large ({}% error)",
                final_energy - initial_energy,
                rel_err * 100.0
            );

            Ok(())
        })();
        match result {
            Ok(()) => {}
            Err(e) if e.is_device_lost() => {}
            Err(e) => panic!("{e}"),
        }
    }
}
