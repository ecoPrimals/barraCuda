// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nosé-Hoover Thermostat
//!
//! **Physics**: Extended Lagrangian for canonical (NVT) ensemble
//! **Properties**: Time-reversible, correctly samples Boltzmann distribution
//! **Use Case**: Production NVT runs (after Berendsen equilibration)
//!
//! **Algorithm**:
//! The Nosé-Hoover thermostat introduces an additional degree of freedom ξ (xi)
//! that acts as a friction coefficient. The equations of motion become:
//!
//! ```text
//! dv/dt = F/m - ξ*v
//! dξ/dt = (1/Q) * (KE - KE_target)
//! ```
//!
//! where Q is the thermostat mass (coupling strength).
//!
//! **Integration** (split for VV compatibility):
//! 1. Half-step ξ: ξ += (dt/2) * (KE - `KE_target`) / Q
//! 2. Half-step v: v' = (v + (dt/2)*a) / (1 + (dt/2)*ξ)  [GPU shader]
//! 3. Full-step x: x += dt * v'                           [integrator]
//! 4. Recompute forces                                    [force kernel]
//! 5. Half-step v: v'' = (v' + (dt/2)*a') / (1 + (dt/2)*ξ) [GPU shader]
//! 6. Half-step ξ: ξ += (dt/2) * (KE'' - `KE_target`) / Q
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ f64 precision

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Nosé-Hoover chain thermostat state
///
/// Maintains the thermostat variable ξ across timesteps.
/// Use `NoseHooverThermostat::new_chain()` to create and
/// `step()` to advance.
pub struct NoseHooverChain {
    /// Thermostat friction coefficient
    pub xi: f64,
    /// Target temperature (reduced units)
    pub t_target: f64,
    /// Thermostat mass Q (coupling strength)
    pub q_mass: f64,
    /// Number of particles (for KE normalization)
    pub n_particles: usize,
    /// Timestep
    pub dt: f64,
}

impl NoseHooverChain {
    /// Create a new Nosé-Hoover chain
    /// # Arguments
    /// * `t_target` - Target temperature in reduced units (T* = 1/Γ for OCP)
    /// * `tau` - Characteristic time for temperature fluctuations
    /// * `n_particles` - Number of particles
    /// * `dt` - Integration timestep
    /// # Notes
    /// The thermostat mass Q is computed from tau: Q = 3*N*`T_target`*tau²
    /// Typical tau values: 10*dt to 100*dt
    #[must_use]
    pub fn new(t_target: f64, tau: f64, n_particles: usize, dt: f64) -> Self {
        // Q = g * k_B * T * tau²  where g = 3N (degrees of freedom)
        // In reduced units k_B = 1
        let g = 3.0 * n_particles as f64;
        let q_mass = g * t_target * tau * tau;

        Self {
            xi: 0.0, // Start with no friction
            t_target,
            q_mass,
            n_particles,
            dt,
        }
    }

    /// Half-step update of thermostat variable ξ
    /// Call this BEFORE and AFTER the velocity half-step.
    /// # Arguments
    /// * `ke_total` - Current total kinetic energy
    pub fn half_step_xi(&mut self, ke_total: f64) {
        // Target KE: KE_target = (3N/2) * T_target (in reduced units, k_B = 1)
        let ke_target = 1.5 * self.n_particles as f64 * self.t_target;

        // dξ/dt = (KE - KE_target) / Q
        // Half-step: ξ += (dt/2) * (KE - KE_target) / Q
        let dxi_dt = (ke_total - ke_target) / self.q_mass;
        self.xi = (0.5 * self.dt).mul_add(dxi_dt, self.xi);
    }

    /// Get current thermostat temperature
    /// Computes the instantaneous temperature that the thermostat
    /// is driving toward.
    #[must_use]
    pub fn current_temperature(&self, ke_total: f64) -> f64 {
        // T = 2*KE / (3N) in reduced units
        2.0 * ke_total / (3.0 * self.n_particles as f64)
    }
}

/// Nosé-Hoover velocity half-step operation
///
/// Applies the thermostat friction to velocities.
/// Use within the split VV integrator:
/// 1. `xi.half_step_xi(KE)`
/// 2. `NoseHooverHalfKick::execute()`  ← this
/// 3. Drift positions
/// 4. Recompute forces
/// 5. `NoseHooverHalfKick::execute()`  ← this
/// 6. `xi.half_step_xi(KE)`
pub struct NoseHooverHalfKick {
    velocities: Tensor,
    forces: Tensor,
    n_particles: usize,
    dt: f64,
    mass: f64,
    xi: f64,
}

impl NoseHooverHalfKick {
    /// Create a new Nosé-Hoover half-kick operation
    /// # Arguments
    /// * `velocities` - Velocity tensor [N, 3] (f64)
    /// * `forces` - Force tensor [N, 3] (f64)
    /// * `dt` - Timestep
    /// * `mass` - Particle mass (3.0 in OCP reduced units)
    /// * `xi` - Current thermostat friction coefficient
    /// # Errors
    /// Returns error if tensor shapes don't match.
    pub fn new(velocities: Tensor, forces: Tensor, dt: f64, mass: f64, xi: f64) -> Result<Self> {
        let vel_shape = velocities.shape();
        if vel_shape.len() != 2 || vel_shape[1] != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 3],
                actual: vel_shape.to_vec(),
            });
        }

        let n_particles = vel_shape[0];

        if forces.shape() != vel_shape {
            return Err(BarracudaError::InvalidShape {
                expected: vel_shape.to_vec(),
                actual: forces.shape().to_vec(),
            });
        }

        Ok(Self {
            velocities,
            forces,
            n_particles,
            dt,
            mass,
            xi,
        })
    }

    /// Execute the half-kick with friction (in-place update)
    /// # Returns
    /// Updated velocities tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.velocities.device();

        // Params: [n, dt, mass, xi, _, _, _, _]
        let params: Vec<f64> = vec![
            self.n_particles as f64,
            self.dt,
            self.mass,
            self.xi,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        let params_bytes: &[u8] = bytemuck::cast_slice(&params);
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NH HalfKick Params"),
                contents: params_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        ComputeDispatch::new(device, "Nosé-Hoover HalfKick")
            .shader(include_str!("nose_hoover.wgsl"), "main")
            .storage_rw(0, self.velocities.buffer())
            .storage_read(1, self.forces.buffer())
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

    #[test]
    fn test_nose_hoover_chain_creation() {
        // OCP at Γ=10: T* = 0.1
        let chain = NoseHooverChain::new(0.1, 0.5, 100, 0.01);

        // Q = 3 * 100 * 0.1 * 0.5² = 7.5
        let expected_q = 3.0 * 100.0 * 0.1 * 0.5 * 0.5;
        assert!(
            (chain.q_mass - expected_q).abs() < 1e-10,
            "Q mass: {} vs {}",
            chain.q_mass,
            expected_q
        );
        assert_eq!(chain.xi, 0.0, "xi should start at 0");
    }

    #[test]
    fn test_nose_hoover_xi_dynamics() {
        let mut chain = NoseHooverChain::new(0.1, 0.5, 100, 0.01);

        // If KE > KE_target (= 1.5 × 100 × 0.1 = 15.0), ξ should increase
        let ke_hot = 20.0; // Too hot

        chain.half_step_xi(ke_hot);
        assert!(chain.xi > 0.0, "xi should increase when too hot");

        // Reset and test cold system
        chain.xi = 0.0;
        let ke_cold = 10.0; // Too cold

        chain.half_step_xi(ke_cold);
        assert!(chain.xi < 0.0, "xi should decrease when too cold");
    }

    #[tokio::test]
    async fn test_nose_hoover_half_kick() {
        let device = crate::device::test_pool::get_test_device().await;

        // Check for f64 support
        if !device
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            return;
        }

        // Single particle
        let velocities: Vec<f64> = vec![1.0, 0.0, 0.0];
        let forces: Vec<f64> = vec![6.0, 0.0, 0.0]; // a = 6/3 = 2

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
        let force_tensor = Tensor::from_buffer(force_buffer, vec![1, 3], device);

        let dt = 0.01;
        let mass = 3.0;
        let xi = 0.1; // Small friction

        let half_kick = NoseHooverHalfKick::new(vel_tensor, force_tensor, dt, mass, xi).unwrap();

        let _vel_after = half_kick.execute().unwrap();
    }
}
