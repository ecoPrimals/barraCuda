// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lennard-Jones Force Calculation
//!
//! **Physics**: Van der Waals interactions (noble gases, simple liquids)
//! **Formula**: F = 24ε/r·[2(σ/r)¹²-(σ/r)⁶]·r̂
//! **Use Case**: Argon, simple MD, coarse-grained models
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Lorentz-Berthelot mixing rules
//! - ✅ Per-particle parameters (agnostic)

use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Lennard-Jones 12-6 potential force calculation
///
/// Computes van der Waals forces between particles.
/// Uses Lorentz-Berthelot mixing rules for multi-component systems.
pub struct LennardJonesForce {
    positions: Tensor, // [N, 3]
    sigmas: Tensor,    // [N] - per-particle σ
    epsilons: Tensor,  // [N] - per-particle ε
    cutoff_radius: f32,
}

impl LennardJonesForce {
    /// Create a Lennard-Jones force calculator with per-particle σ and ε parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if positions shape is not [N, 3], or sigmas/epsilons shape does not match.
    pub fn new(
        positions: Tensor,
        sigmas: Tensor,
        epsilons: Tensor,
        cutoff_radius: Option<f32>,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];

        // Validate sigmas and epsilons
        for tensor in [&sigmas, &epsilons] {
            let shape = tensor.shape();
            if shape.len() != 1 || shape[0] != n_particles {
                return Err(BarracudaError::InvalidShape {
                    expected: vec![n_particles],
                    actual: shape.to_vec(),
                });
            }
        }

        Ok(Self {
            positions,
            sigmas,
            epsilons,
            cutoff_radius: cutoff_radius.unwrap_or(2.5), // Standard LJ cutoff
        })
    }

    /// Compute Lennard-Jones forces for all particles and return the force tensor [N, 3].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        let output_size = (n_particles * 3 * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LJ Forces Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            cutoff_radius: f32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            cutoff_radius: self.cutoff_radius,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LJ Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups = (n_particles as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(device, "LennardJones")
            .shader(include_str!("lennard_jones.wgsl"), "main")
            .storage_read(0, self.positions.buffer())
            .storage_read(1, self.sigmas.buffer())
            .storage_read(2, self.epsilons.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![n_particles, 3],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lj_force_argon() {
        let device = crate::device::test_pool::get_test_device().await;

        // Two argon atoms
        let positions = vec![0.0, 0.0, 0.0, 3.4, 0.0, 0.0]; // σ apart
        let sigmas = vec![3.4, 3.4]; // Argon σ (Angstroms)
        let epsilons = vec![1.0, 1.0]; // Normalized

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let sigma_tensor = Tensor::from_data(&sigmas, vec![2], device.clone()).unwrap();
        let epsilon_tensor = Tensor::from_data(&epsilons, vec![2], device).unwrap();

        let lj = LennardJonesForce::new(pos_tensor, sigma_tensor, epsilon_tensor, None).unwrap();
        let forces = lj.execute().unwrap();

        assert_eq!(forces.shape(), &[2, 3]);
    }
}
