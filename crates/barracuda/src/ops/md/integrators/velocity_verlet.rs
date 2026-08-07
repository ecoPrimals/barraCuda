// SPDX-License-Identifier: AGPL-3.0-or-later
//! Velocity-Verlet Time Integration
//!
//! **Physics**: Symplectic integrator for classical MD
//! **Properties**: Energy-conserving, time-reversible, 2nd-order accurate
//! **Use Case**: Molecular dynamics, planetary motion
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader  
//! - ✅ Zero unsafe code
//! - ✅ Agnostic (no hardcoded system)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Velocity-Verlet time integration
///
/// Updates positions and velocities for one time step.
/// Requires forces at both t and t+Δt.
pub struct VelocityVerlet {
    positions: Tensor,  // [N, 3]
    velocities: Tensor, // [N, 3]
    forces_old: Tensor, // [N, 3] at time t
    forces_new: Tensor, // [N, 3] at time t+Δt
    masses: Tensor,     // [N]
    dt: f32,
}

impl VelocityVerlet {
    /// Create Velocity-Verlet integrator with given state tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if tensor shapes mismatch, or dt <= 0.
    pub fn new(
        positions: Tensor,
        velocities: Tensor,
        forces_old: Tensor,
        forces_new: Tensor,
        masses: Tensor,
        dt: f32,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate all tensors have matching shapes
        for tensor in [&velocities, &forces_old, &forces_new] {
            if tensor.shape() != pos_shape {
                return Err(BarracudaError::InvalidShape {
                    expected: pos_shape.to_vec(),
                    actual: tensor.shape().to_vec(),
                });
            }
        }

        let mass_shape = masses.shape();
        if mass_shape.len() != 1 || mass_shape[0] != n_particles {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n_particles],
                actual: mass_shape.to_vec(),
            });
        }

        if dt <= 0.0 {
            return Err(BarracudaError::Device(
                "Time step dt must be positive".to_string(),
            ));
        }

        Ok(Self {
            positions,
            velocities,
            forces_old,
            forces_new,
            masses,
            dt,
        })
    }

    /// Execute Velocity-Verlet integration
    ///
    /// # Returns
    /// (`positions_new`, `velocities_new`) at time t+Δt
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        // Create output buffers
        let buffer_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;

        let positions_new_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VV Positions New"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocities_new_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VV Velocities New"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            dt: f32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            dt: self.dt,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VV Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "Velocity-Verlet")
            .shader(include_str!("velocity_verlet.wgsl"), "main")
            .storage_read(0, self.positions.buffer())
            .storage_read(1, self.velocities.buffer())
            .storage_read(2, self.forces_old.buffer())
            .storage_read(3, self.forces_new.buffer())
            .storage_read(4, self.masses.buffer())
            .storage_rw(5, &positions_new_buffer)
            .storage_rw(6, &velocities_new_buffer)
            .uniform(7, &params_buffer)
            .dispatch_1d(n_particles as u32)
            .submit()?;

        let positions_new =
            Tensor::from_buffer(positions_new_buffer, vec![n_particles, 3], device.clone());

        let velocities_new =
            Tensor::from_buffer(velocities_new_buffer, vec![n_particles, 3], device.clone());

        Ok((positions_new, velocities_new))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_velocity_verlet_single_particle() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Single particle in free space with constant force
        let positions = vec![0.0, 0.0, 0.0];
        let velocities = vec![1.0, 0.0, 0.0]; // v_x = 1
        let forces_old = vec![2.0, 0.0, 0.0]; // F = 2
        let forces_new = vec![2.0, 0.0, 0.0]; // Constant force
        let masses = vec![1.0]; // m = 1
        let dt = 0.1;

        let pos_tensor = Tensor::from_data(&positions, vec![1, 3], device.clone()).unwrap();
        let vel_tensor = Tensor::from_data(&velocities, vec![1, 3], device.clone()).unwrap();
        let f_old_tensor = Tensor::from_data(&forces_old, vec![1, 3], device.clone()).unwrap();
        let f_new_tensor = Tensor::from_data(&forces_new, vec![1, 3], device.clone()).unwrap();
        let mass_tensor = Tensor::from_data(&masses, vec![1], device).unwrap();

        // Verify inputs are correct
        let pos_check = pos_tensor.to_vec().unwrap();
        let vel_check = vel_tensor.to_vec().unwrap();
        let f_old_check = f_old_tensor.to_vec().unwrap();
        let f_new_check = f_new_tensor.to_vec().unwrap();
        let mass_check = mass_tensor.to_vec().unwrap();

        assert_eq!(pos_check, positions);
        assert_eq!(vel_check, velocities);
        assert_eq!(f_old_check, forces_old);
        assert_eq!(f_new_check, forces_new);
        assert_eq!(mass_check, masses);

        let vv = VelocityVerlet::new(
            pos_tensor,
            vel_tensor,
            f_old_tensor,
            f_new_tensor,
            mass_tensor,
            dt,
        )
        .unwrap();

        let (pos_new, vel_new) = vv.execute().unwrap();

        let pos_data = pos_new.to_vec().unwrap();
        let vel_data = vel_new.to_vec().unwrap();

        // Check physics: x = x0 + v*t + 0.5*a*t^2
        // x = 0 + 1*0.1 + 0.5*2*0.01 = 0.11
        let expected_x = (0.5 * 2.0 * dt).mul_add(dt, 1.0f32.mul_add(dt, 0.0));
        assert!((pos_data[0] - expected_x).abs() < 1e-5, "Position update");

        // v = v0 + a*t
        // v = 1 + 2*0.1 = 1.2
        let expected_v = 2.0f32.mul_add(dt, 1.0);
        assert!((vel_data[0] - expected_v).abs() < 1e-5, "Velocity update");
    }
}
