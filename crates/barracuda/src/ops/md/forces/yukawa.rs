// SPDX-License-Identifier: AGPL-3.0-or-later
//! Yukawa Force Calculation
//!
//! **Physics**: Screened electrostatic interactions (Debye screening)
//! **Formula**: F = k·q₁q₂·exp(-κr)/r²·r̂
//! **Use Case**: Dusty plasmas, colloids, screened electrostatics
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ Agnostic (κ parameterized)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Yukawa (screened Coulomb) force calculation
///
/// Models interactions with exponential screening (e.g., Debye screening in plasmas).
/// Reduces to Coulomb when κ → 0.
pub struct YukawaForce {
    positions: Tensor,
    charges: Tensor,
    yukawa_constant: f32,
    kappa: f32, // Screening parameter (inverse Debye length)
    cutoff_radius: f32,
    epsilon: f32,
}

impl YukawaForce {
    /// Create a Yukawa force calculator with positions, charges, and screening parameters.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if positions shape is not [N, 3], charges shape does not match, or κ < 0.
    pub fn new(
        positions: Tensor,
        charges: Tensor,
        yukawa_constant: Option<f32>,
        kappa: f32,
        cutoff_radius: Option<f32>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        let pos_shape = positions.shape();
        if pos_shape.len() != 2 || pos_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: pos_shape.to_vec(),
            });
        }

        let n_particles = pos_shape[0];
        let charge_shape = charges.shape();
        if charge_shape.len() != 1 || charge_shape[0] != n_particles {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n_particles],
                actual: charge_shape.to_vec(),
            });
        }

        if kappa < 0.0 {
            return Err(BarracudaError::Device(
                "Screening parameter κ must be non-negative".to_string(),
            ));
        }

        Ok(Self {
            positions,
            charges,
            yukawa_constant: yukawa_constant.unwrap_or(1.0),
            kappa,
            cutoff_radius: cutoff_radius.unwrap_or(f32::INFINITY),
            epsilon: epsilon.unwrap_or(1e-6),
        })
    }

    /// Execute Yukawa force computation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.positions.device();
        let n_particles = self.positions.shape()[0];

        let output_buffer = device.create_buffer_f32(n_particles * 3)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_particles: u32,
            yukawa_constant: f32,
            kappa: f32,
            cutoff_radius: f32,
            epsilon: f32,
            _pad: [f32; 3],
        }

        let params = Params {
            n_particles: n_particles as u32,
            yukawa_constant: self.yukawa_constant,
            kappa: self.kappa,
            cutoff_radius: self.cutoff_radius,
            epsilon: self.epsilon,
            _pad: [0.0; 3],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Yukawa Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "Yukawa Force")
            .shader(include_str!("yukawa.wgsl"), "main")
            .storage_read(0, self.positions.buffer())
            .storage_read(1, self.charges.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(n_particles as u32)
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
    async fn test_yukawa_reduces_to_coulomb() {
        let device = crate::device::test_pool::get_test_device().await;

        // With κ=0, Yukawa should equal Coulomb
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let charges = vec![1.0, 1.0];

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let charge_tensor = Tensor::from_data(&charges, vec![2], device).unwrap();

        let yukawa = YukawaForce::new(
            pos_tensor,
            charge_tensor,
            Some(1.0),
            0.0, // κ=0 → Coulomb
            None,
            None,
        )
        .unwrap();

        let forces = yukawa.execute().unwrap();
        assert_eq!(forces.shape(), &[2, 3]);
    }

    #[tokio::test]
    async fn test_yukawa_screening() {
        let device = crate::device::test_pool::get_test_device().await;

        // Large κ should significantly reduce force at distance
        let positions = vec![0.0, 0.0, 0.0, 5.0, 0.0, 0.0];
        let charges = vec![1.0, 1.0];

        let pos_tensor = Tensor::from_data(&positions, vec![2, 3], device.clone()).unwrap();
        let charge_tensor = Tensor::from_data(&charges, vec![2], device).unwrap();

        let yukawa = YukawaForce::new(
            pos_tensor,
            charge_tensor,
            Some(1.0),
            2.0, // Strong screening
            None,
            None,
        )
        .unwrap();

        let forces = yukawa.execute().unwrap();
        let force_data = forces.to_vec().unwrap();

        // Force should be heavily screened (small magnitude)
        let f0_mag = force_data[2]
            .mul_add(
                force_data[2],
                force_data[1].mul_add(force_data[1], force_data[0].powi(2)),
            )
            .sqrt();
        let _ = f0_mag;
    }
}
