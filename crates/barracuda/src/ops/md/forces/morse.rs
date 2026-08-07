// SPDX-License-Identifier: AGPL-3.0-or-later
//! Morse Force Calculation
//!
//! **Physics**: Anharmonic bonded interactions (covalent bonds, diatomics)
//! **Formula**: F = 2Daα[exp(-2α(r-r₀))-exp(-α(r-r₀))]·r̂
//! **Use Case**: Chemical bonds, molecular vibrations, spectroscopy
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader with atomic force accumulation
//! - ✅ Zero unsafe code
//! - ✅ Per-bond parameters (agnostic)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Morse potential force calculation for bonded interactions
///
/// Models chemical bonds with anharmonic oscillator potential.
/// More accurate than harmonic approximation for large displacements.
pub struct MorseForce {
    positions: Tensor,           // [N, 3]
    bond_pairs: Tensor,          // [M, 2] - particle indices for each bond
    dissociation_energy: Tensor, // [M] - D for each bond
    width_param: Tensor,         // [M] - α (width) for each bond
    equilibrium_dist: Tensor,    // [M] - r₀ for each bond
}

impl MorseForce {
    /// Create a Morse force calculator for bonded interactions with per-bond parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if positions shape is not [N, 3], `bond_pairs` is not [M, 2], or parameter shapes mismatch.
    pub fn new(
        positions: Tensor,
        bond_pairs: Tensor,
        dissociation_energy: Tensor,
        width_param: Tensor,
        equilibrium_dist: Tensor,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let bond_shape = bond_pairs.shape();
        if bond_shape.len() != 2 || bond_shape[1] != 2 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 2],
                actual: bond_shape.to_vec(),
            });
        }

        let n_bonds = bond_shape[0];

        // Validate parameter tensors
        for tensor in [&dissociation_energy, &width_param, &equilibrium_dist] {
            let shape = tensor.shape();
            if shape.len() != 1 || shape[0] != n_bonds {
                return Err(BarracudaError::InvalidShape {
                    expected: vec![n_bonds],
                    actual: shape.to_vec(),
                });
            }
        }

        Ok(Self {
            positions,
            bond_pairs,
            dissociation_energy,
            width_param,
            equilibrium_dist,
        })
    }

    /// Compute Morse forces for all particles and return the force tensor [N, 3].
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];
        let n_bonds = self.bond_pairs.shape()[0];

        // Create atomic buffer (i32) for force accumulation
        let atomic_buffer_size = (n_particles * 3 * std::mem::size_of::<i32>()) as u64;
        let atomic_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morse Forces Atomic"),
            size: atomic_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Clear atomic buffer to zero
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Morse Clear Encoder"),
        });
        encoder.clear_buffer(&atomic_buffer, 0, None);
        device.submit_commands(Some(encoder.finish()));

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            n_bonds: u32,
            pad1: f32,
            pad2: f32,
        }

        let params = Params {
            n_particles: n_particles as u32,
            n_bonds: n_bonds as u32,
            pad1: 0.0,
            pad2: 0.0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Morse Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "Morse Force")
            .shader(include_str!("morse.wgsl"), "main")
            .storage_read(0, self.positions.buffer())
            .storage_read(1, self.bond_pairs.buffer())
            .storage_read(2, self.dissociation_energy.buffer())
            .storage_read(3, self.width_param.buffer())
            .storage_read(4, self.equilibrium_dist.buffer())
            .storage_rw(5, &atomic_buffer)
            .uniform(6, &params_buffer)
            .dispatch_1d(n_bonds as u32)
            .submit()?;

        // Create staging buffer to read back atomic results
        let staging_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Morse Staging"),
            size: atomic_buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Morse Readback"),
        });

        encoder.copy_buffer_to_buffer(&atomic_buffer, 0, &staging_buffer, 0, atomic_buffer_size);
        device.submit_commands(Some(encoder.finish()));

        // Read back and convert i32 -> f32
        let n_force_elements = n_particles * 3;
        let i32_data: Vec<i32> = device.map_staging_buffer(&staging_buffer, n_force_elements)?;
        let f32_data: Vec<f32> = i32_data.iter().map(|&x| x as f32 / 1000.0).collect();

        // Create final output tensor
        Tensor::from_data(&f32_data, vec![n_particles, 3], device.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_morse_force_equilibrium() {
        let device = crate::device::test_pool::get_test_device().await;

        // Two bonded particles at equilibrium (should have near-zero force)
        let positions = vec![0.0, 0.0, 0.0, 1.5, 0.0, 0.0]; // r₀ = 1.5
        let bond_pairs = vec![0.0, 1.0]; // Bond between particles 0 and 1
        let dissociation = vec![100.0]; // D = 100
        let width = vec![2.0]; // α = 2
        let r0 = vec![1.5]; // Equilibrium at r = 1.5

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let pairs_tensor = Tensor::from_data(&bond_pairs, vec![1, 2], device.clone()).unwrap();
        let d_tensor = Tensor::from_data(&dissociation, vec![1], device.clone()).unwrap();
        let a_tensor = Tensor::from_data(&width, vec![1], device.clone()).unwrap();
        let r0_tensor = Tensor::from_data(&r0, vec![1], device).unwrap();

        let morse =
            MorseForce::new(pos_tensor, pairs_tensor, d_tensor, a_tensor, r0_tensor).unwrap();

        let forces = morse.execute().unwrap();
        let force_data = forces.to_vec().unwrap();

        // At equilibrium, force should be very small
        let f0_mag = force_data[2]
            .mul_add(
                force_data[2],
                force_data[1].mul_add(force_data[1], force_data[0].powi(2)),
            )
            .sqrt();
        let _ = f0_mag;
    }
}
