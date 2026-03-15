// SPDX-License-Identifier: AGPL-3.0-only
//! KINETIC ENERGY F64 — Per-particle and total kinetic energy — GPU shader dispatch.
//!
//! All math originates as `kinetic_energy_f64.wgsl`.
//!
//! Applications:
//! - Temperature calculation: T = 2*`KE_total` / (3*N*`k_B`)
//! - Energy monitoring in MD
//! - Thermostat validation

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("kinetic_energy_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct KeParams {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated f64 kinetic energy calculator.
pub struct KineticEnergyF64 {
    device: Arc<WgpuDevice>,
}

impl KineticEnergyF64 {
    /// Creates a new kinetic energy calculator for the given WGPU device.
    /// # Errors
    /// Returns [`Err`] if device initialization fails (currently always succeeds).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute per-particle KE on GPU: `KE_i` = ½ `m_i` `v_i²`
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, staging buffer mapping fails, or the device is lost.
    pub fn per_particle(&self, velocities: &[f64], masses: &[f64]) -> Result<Vec<f64>> {
        let n = masses.len();
        let d = &self.device.device;

        let vel_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KE:vel"),
            contents: bytemuck::cast_slice(velocities),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mass_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KE:mass"),
            contents: bytemuck::cast_slice(masses),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let out_size = (n * 8) as u64;
        let ke_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KE:out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = KeParams {
            n_particles: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("KE:params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wg_count = (n as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(&self.device, "kinetic_energy_f64")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &vel_buf)
            .storage_read(1, &mass_buf)
            .storage_rw(2, &ke_buf)
            .uniform(3, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit()?;

        let rb = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KE:rb"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self.device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&ke_buf, 0, &rb, 0, out_size);
        self.device.submit_commands(Some(enc.finish()));

        self.device.map_staging_buffer::<f64>(&rb, n)
    }

    /// Compute total kinetic energy (GPU per-particle, host reduce).
    /// # Errors
    /// Returns [`Err`] if [`per_particle`](Self::per_particle) fails.
    pub fn total(&self, velocities: &[f64], masses: &[f64]) -> Result<f64> {
        let per_particle = self.per_particle(velocities, masses)?;
        Ok(per_particle.iter().sum())
    }

    /// Compute temperature: T = 2*`KE_total` / (3*N*`k_B`).
    /// # Errors
    /// Returns [`Err`] if [`total`](Self::total) fails.
    pub fn temperature(&self, velocities: &[f64], masses: &[f64], k_b: f64) -> Result<f64> {
        let n = masses.len();
        if n == 0 {
            return Ok(0.0);
        }
        let ke_total = self.total(velocities, masses)?;
        Ok(2.0 * ke_total / (3.0 * n as f64 * k_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_single_particle() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let ke_calc = KineticEnergyF64::new(device)?;

        let velocities = vec![1.0, 0.0, 0.0];
        let masses = vec![2.0];

        let per_particle = ke_calc.per_particle(&velocities, &masses)?;

        // KE = ½mv² = ½ * 2 * 1² = 1
        assert!((per_particle[0] - 1.0).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_3d_velocity() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let ke_calc = KineticEnergyF64::new(device)?;

        let velocities = vec![1.0, 2.0, 2.0]; // |v| = 3
        let masses = vec![1.0];

        let per_particle = ke_calc.per_particle(&velocities, &masses)?;

        // KE = ½ * 1 * 9 = 4.5
        assert!((per_particle[0] - 4.5).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_total_energy() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let ke_calc = KineticEnergyF64::new(device)?;

        let velocities = vec![
            1.0, 0.0, 0.0, // Particle 0
            0.0, 2.0, 0.0, // Particle 1
        ];
        let masses = vec![1.0, 1.0];

        let total = ke_calc.total(&velocities, &masses)?;

        // KE = ½*1*1 + ½*1*4 = 0.5 + 2.0 = 2.5
        assert!((total - 2.5).abs() < 1e-10);

        Ok(())
    }

    #[test]
    fn test_temperature() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let ke_calc = KineticEnergyF64::new(device)?;

        // Ideal gas: equipartition gives each DOF ½k_B T
        // For N particles in 3D: KE = 3/2 N k_B T
        // So T = 2 KE / (3 N k_B)

        let n = 100;
        let k_b = 1.0;
        let target_temp = 300.0;
        let target_ke_per_particle = 1.5 * k_b * target_temp;
        let v_rms = (2.0_f64 * target_ke_per_particle).sqrt(); // For m=1

        // Create velocities with RMS speed
        let mut velocities = vec![0.0; n * 3];
        for i in 0..n {
            // Distribute velocity isotropically (simplified)
            let scale = v_rms / 3.0_f64.sqrt();
            velocities[i * 3] = scale;
            velocities[i * 3 + 1] = scale;
            velocities[i * 3 + 2] = scale;
        }
        let masses = vec![1.0; n];

        let temp = ke_calc.temperature(&velocities, &masses, k_b)?;

        let rel_err = (temp - target_temp).abs() / target_temp;
        assert!(
            rel_err < 0.01,
            "Temperature {temp} not close to {target_temp}"
        );

        Ok(())
    }
}
