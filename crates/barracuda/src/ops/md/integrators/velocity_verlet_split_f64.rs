// SPDX-License-Identifier: AGPL-3.0-or-later
//! Split Velocity-Verlet Integrator (f64)
//!
//! **Algorithm**: Kick-drift-kick pattern for flexible thermostating
//! **Precision**: Full f64 via `math_f64.wgsl` preamble
//! **Reference**: Standard in LAMMPS, GROMACS
//!
//! **Advantages over monolithic VV**:
//! - Thermostat can be applied between kicks
//! - Force kernel can be swapped without touching integrator
//! - Explicit PBC wrapping during drift
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader (`velocity_verlet_split.wgsl`, `vv_half_kick_f64.wgsl`)
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Split Velocity-Verlet Step 1: Half-kick + drift + PBC wrap
///
/// Updates velocities by half-step and positions by full step.
/// PBC wrapping applied during drift to keep positions in [0, box).
pub struct VelocityVerletKickDrift {
    positions: Tensor,
    velocities: Tensor,
    forces: Tensor,
    n_particles: usize,
    dt: f64,
    mass: f64,
    box_size: [f64; 3],
}

impl VelocityVerletKickDrift {
    /// Create a new kick-drift operation
    /// # Arguments
    /// * `positions` - Position tensor [N, 3] (f64)
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `forces` - Force tensor [N, 3] (f64)
    /// * `dt` - Timestep (reduced units)
    /// * `mass` - Particle mass (3.0 in OCP reduced units)
    /// * `box_size` - Simulation box dimensions [Lx, Ly, Lz]
    /// # Errors
    /// Returns error if tensor shapes don't match.
    pub fn new(
        positions: Tensor,
        velocities: Tensor,
        forces: Tensor,
        dt: f64,
        mass: f64,
        box_size: [f64; 3],
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate matching shapes
        if velocities.shape() != pos_shape || forces.shape() != pos_shape {
            return Err(BarracudaError::InvalidShape {
                expected: pos_shape.to_vec(),
                actual: velocities.shape().to_vec(),
            });
        }

        if dt <= 0.0 {
            return Err(BarracudaError::Device(
                "Timestep dt must be positive".to_string(),
            ));
        }

        Ok(Self {
            positions,
            velocities,
            forces,
            n_particles,
            dt,
            mass,
            box_size,
        })
    }

    /// Execute the kick-drift step (in-place update)
    /// # Returns
    /// (positions, velocities) after half-kick and drift
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.positions.device();

        // Params: [n, dt, mass, _, box_x, box_y, box_z, _]
        let params: Vec<f64> = vec![
            self.n_particles as f64,
            self.dt,
            self.mass,
            0.0,
            self.box_size[0],
            self.box_size[1],
            self.box_size[2],
            0.0,
        ];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VV KickDrift Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        ComputeDispatch::new(device, "VV KickDrift")
            .shader(include_str!("velocity_verlet_split.wgsl"), "main")
            .f64()
            .storage_rw(0, self.positions.buffer())
            .storage_rw(1, self.velocities.buffer())
            .storage_read(2, self.forces.buffer())
            .storage_read(3, &params_buffer)
            .dispatch(
                (self.n_particles as u32).div_ceil(WORKGROUP_SIZE_COMPACT),
                1,
                1,
            )
            .submit()?;

        Ok((self.positions, self.velocities))
    }
}

/// Split Velocity-Verlet Step 3: Second half-kick
///
/// Completes the velocity update using NEW forces (after drift).
pub struct VelocityVerletHalfKick {
    velocities: Tensor,
    forces_new: Tensor,
    n_particles: usize,
    dt: f64,
    mass: f64,
}

impl VelocityVerletHalfKick {
    /// Create a new half-kick operation
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64) — after kick-drift
    /// * `forces_new` - Force tensor [N, 3] (f64) — recomputed after drift
    /// * `dt` - Timestep (reduced units)
    /// * `mass` - Particle mass (3.0 in OCP reduced units)
    /// # Errors
    /// Returns error if tensor shapes don't match.
    pub fn new(velocities: Tensor, forces_new: Tensor, dt: f64, mass: f64) -> Result<Self> {
        let vel_shape = velocities.shape();
        if vel_shape.len() != 2 || vel_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: vel_shape.to_vec(),
            });
        }

        let n_particles = vel_shape[0];

        if forces_new.shape() != vel_shape {
            return Err(BarracudaError::InvalidShape {
                expected: vel_shape.to_vec(),
                actual: forces_new.shape().to_vec(),
            });
        }

        Ok(Self {
            velocities,
            forces_new,
            n_particles,
            dt,
            mass,
        })
    }

    /// Execute the second half-kick (in-place update)
    /// # Returns
    /// Velocities after full VV step
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();

        // Params: [n, dt, mass, _]
        let params: Vec<f64> = vec![self.n_particles as f64, self.dt, self.mass, 0.0];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VV HalfKick Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        ComputeDispatch::new(device, "VV HalfKick")
            .shader(include_str!("vv_half_kick_f64.wgsl"), "main")
            .f64()
            .storage_rw(0, self.velocities.buffer())
            .storage_read(1, self.forces_new.buffer())
            .storage_read(2, &params_buffer)
            .dispatch(
                (self.n_particles as u32).div_ceil(WORKGROUP_SIZE_COMPACT),
                1,
                1,
            )
            .submit()?;

        Ok(self.velocities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_split_vv_single_particle() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // Check for f64 support
        if !device
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            return;
        }

        // Single particle with constant force
        let positions: Vec<f64> = vec![0.0, 0.0, 0.0];
        let velocities: Vec<f64> = vec![1.0, 0.0, 0.0];
        let forces: Vec<f64> = vec![6.0, 0.0, 0.0]; // F = 6 → a = 6/3 = 2

        let dt = 0.1;
        let mass = 3.0; // OCP reduced units
        let box_size = [10.0, 10.0, 10.0];

        // Create tensors
        let pos_bytes: &[u8] = bytemuck::cast_slice(&positions);
        let pos_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Positions"),
                contents: pos_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let pos_tensor = Tensor::from_buffer(pos_buffer, vec![1, 3], device.clone());

        let vel_bytes: &[u8] = bytemuck::cast_slice(&velocities);
        let vel_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Velocities"),
                contents: vel_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let vel_tensor = Tensor::from_buffer(vel_buffer, vec![1, 3], device.clone());

        let force_bytes: &[u8] = bytemuck::cast_slice(&forces);
        let force_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Forces"),
                contents: force_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let force_tensor = Tensor::from_buffer(force_buffer, vec![1, 3], device.clone());

        // Step 1: Kick-drift
        let kick_drift =
            VelocityVerletKickDrift::new(pos_tensor, vel_tensor, force_tensor, dt, mass, box_size)
                .unwrap();

        let (_pos_after, vel_after) = kick_drift.execute().unwrap();

        // Step 2 would be force recomputation (skipped here)
        // Step 3: Second half-kick (using same forces as approximation)
        let force_bytes2: &[u8] = bytemuck::cast_slice(&forces);
        let force_buffer2 = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Forces 2"),
                contents: force_bytes2,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let force_tensor2 = Tensor::from_buffer(force_buffer2, vec![1, 3], device);

        let half_kick = VelocityVerletHalfKick::new(vel_after, force_tensor2, dt, mass).unwrap();
        let _vel_final = half_kick.execute().unwrap();
    }
}
