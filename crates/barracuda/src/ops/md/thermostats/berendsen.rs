// SPDX-License-Identifier: AGPL-3.0-or-later
//! Berendsen Thermostat
//!
//! **Physics**: Weak coupling to heat bath via velocity rescaling
//! **Formula**: v *= sqrt(1 + (dt/τ) * (`T_target/T_current` - 1))
//! **Use Case**: Equilibration phase only — does NOT sample canonical ensemble
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Berendsen velocity rescaling thermostat
///
/// The scale factor is computed on CPU from current temperature:
/// ```text
/// scale = sqrt(1 + (dt/tau) * (T_target/T_current - 1))
/// ```
/// Then applied uniformly to all velocities.
pub struct BerendsenThermostat {
    velocities: Tensor,
    scale_factor: f64,
}

impl BerendsenThermostat {
    /// Create a new Berendsen thermostat operation
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `scale_factor` - Pre-computed scale factor from temperature ratio
    /// # Errors
    /// Returns error if velocities tensor has wrong shape.
    pub fn new(velocities: Tensor, scale_factor: f64) -> Result<Self> {
        let shape = velocities.shape();
        if shape.len() != 2 || shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: shape.to_vec(),
            });
        }

        Ok(Self {
            velocities,
            scale_factor,
        })
    }

    /// Compute the Berendsen scale factor
    /// # Arguments
    /// * `t_current` - Current temperature (reduced units)
    /// * `t_target` - Target temperature (reduced units)
    /// * `dt` - Timestep (reduced units)
    /// * `tau` - Coupling time constant (reduced units)
    #[must_use]
    pub fn compute_scale(t_current: f64, t_target: f64, dt: f64, tau: f64) -> f64 {
        if t_current < 1e-30 {
            return 1.0; // avoid division by zero
        }
        let ratio = (dt / tau).mul_add(t_target / t_current - 1.0, 1.0);
        ratio.max(0.0).sqrt()
    }

    /// Execute the thermostat (in-place velocity scaling)
    /// # Returns
    /// The same velocities tensor with scaled values
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();
        let n_particles = self.velocities.shape()[0];

        // Create params buffer: [n, scale, _, _]
        let params: Vec<f64> = vec![n_particles as f64, self.scale_factor, 0.0, 0.0];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Berendsen Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        ComputeDispatch::new(device, "Berendsen")
            .shader(include_str!("berendsen.wgsl"), "main")
            .storage_rw(0, self.velocities.buffer())
            .storage_read(1, &params_buffer)
            .dispatch_1d(n_particles as u32)
            .submit()?;

        Ok(self.velocities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_berendsen_scale_computation() {
        // T_current = 0.01, T_target = 0.00633, dt = 0.01, tau = 0.05
        // Expected: scale = sqrt(1 + 0.2 * (0.633 - 1)) ≈ sqrt(0.9266) ≈ 0.9626
        let scale = BerendsenThermostat::compute_scale(0.01, 0.00633, 0.01, 0.05);
        assert!((scale - 0.9626).abs() < 0.01);
    }
}
