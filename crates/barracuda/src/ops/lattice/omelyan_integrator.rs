// SPDX-License-Identifier: AGPL-3.0-or-later

//! Omelyan 2MN (second-order minimum-norm) integrator for HMC.
//!
//! Composes the existing `GpuHmcLeapfrog` `momentum_kick` and `link_update`
//! kernels into the Omelyan stepping pattern:
//!
//!   π(λε) → U(ε/2) → π((1-2λ)ε) → U(ε/2) → π(λε)
//!
//! where λ ≈ 0.1932 is the optimal 2MN parameter (Omelyan, Mryglod,
//! Folk, 2003). This achieves O(ε⁴) energy conservation vs O(ε²) for
//! standard leapfrog, enabling larger step sizes at the same acceptance.
//!
//! Provenance: hotSpring/wateringHole V69 request → toadStool absorption.
//!
//! # References
//! Omelyan, Mryglod, Folk (2003) Comp. Phys. Comm. 146, 188

use super::gpu_hmc_leapfrog::GpuHmcLeapfrog;
use crate::error::Result;

/// Optimal λ for the 2MN Omelyan integrator (Omelyan et al. 2003, Eq. 31).
pub const OMELYAN_LAMBDA: f64 = 0.193_183_3;

/// Omelyan 2MN integrator wrapping existing GPU leapfrog kernels.
///
/// Each `step()` call performs one full Omelyan step of size `dt`,
/// using 3 momentum kicks and 2 link updates (5 GPU dispatches).
///
/// For a trajectory of length τ with N steps: dt = τ/N, call `step()` N times.
pub struct OmelyanIntegrator {
    leapfrog: GpuHmcLeapfrog,
    lambda: f64,
}

impl OmelyanIntegrator {
    /// Create with the standard optimal λ.
    #[must_use]
    pub fn new(leapfrog: GpuHmcLeapfrog) -> Self {
        Self {
            leapfrog,
            lambda: OMELYAN_LAMBDA,
        }
    }

    /// Create with a custom λ (for studies of integrator tuning).
    #[must_use]
    pub fn with_lambda(leapfrog: GpuHmcLeapfrog, lambda: f64) -> Self {
        Self { leapfrog, lambda }
    }

    /// Execute one full Omelyan 2MN step of size `dt`.
    /// Sequence: π(λ·dt) → U(dt/2) → π((1-2λ)·dt) → U(dt/2) → π(λ·dt)
    /// The force buffer must contain the current gauge force (computed
    /// externally between steps via `GpuHmcForceSu3` or equivalent).
    /// For a complete HMC trajectory, the caller should:
    /// 1. Compute force
    /// 2. Call `step()`  
    /// 3. Recompute force
    /// 4. Repeat N times
    ///
    /// With force reuse at boundaries, the effective cost per step is
    /// 1 force computation + 5 kernel dispatches.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn step(
        &self,
        links_buf: &wgpu::Buffer,
        momenta_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        dt: f64,
    ) -> Result<()> {
        let lam = self.lambda;

        // Step 1: half-kick with λε
        self.leapfrog.momentum_kick(
            links_buf,
            momenta_buf,
            force_buf,
            rng_buf,
            volume,
            lam * dt,
        )?;

        // Step 2: full position update with ε/2
        self.leapfrog
            .link_update(links_buf, momenta_buf, force_buf, rng_buf, volume, dt * 0.5)?;

        // Step 3: central kick with (1-2λ)ε
        self.leapfrog.momentum_kick(
            links_buf,
            momenta_buf,
            force_buf,
            rng_buf,
            volume,
            (1.0 - 2.0 * lam) * dt,
        )?;

        // Step 4: full position update with ε/2
        self.leapfrog
            .link_update(links_buf, momenta_buf, force_buf, rng_buf, volume, dt * 0.5)?;

        // Step 5: half-kick with λε
        self.leapfrog.momentum_kick(
            links_buf,
            momenta_buf,
            force_buf,
            rng_buf,
            volume,
            lam * dt,
        )?;

        Ok(())
    }

    /// Execute a multi-step trajectory (N steps of size dt = `trajectory_length` / `n_steps`).
    /// This assumes the force is constant across all steps (quenched approximation
    /// or external force recomputation via callback).
    /// For dynamical fermion HMC where force must be recomputed each step,
    /// use `step()` directly in a loop with force recomputation between steps.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn trajectory_quenched(
        &self,
        links_buf: &wgpu::Buffer,
        momenta_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        trajectory_length: f64,
        n_steps: usize,
    ) -> Result<()> {
        let dt = trajectory_length / n_steps as f64;
        for _ in 0..n_steps {
            self.step(links_buf, momenta_buf, force_buf, rng_buf, volume, dt)?;
        }
        Ok(())
    }

    /// Access the underlying leapfrog for momentum generation.
    #[must_use]
    pub fn leapfrog(&self) -> &GpuHmcLeapfrog {
        &self.leapfrog
    }

    /// Current λ parameter.
    #[must_use]
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omelyan_lambda_correct() {
        assert!((OMELYAN_LAMBDA - 0.1931833).abs() < 1e-7);
    }

    #[test]
    fn omelyan_step_sizes_sum_to_dt() {
        let lam = OMELYAN_LAMBDA;
        let dt = 0.1;
        let kick_total = lam * dt + (1.0 - 2.0 * lam) * dt + lam * dt;
        assert!(
            (kick_total - dt).abs() < 1e-15,
            "Kick sum {kick_total} != dt {dt}"
        );
        let update_total = dt * 0.5 + dt * 0.5;
        assert!(
            (update_total - dt).abs() < 1e-15,
            "Update sum {update_total} != dt {dt}"
        );
    }

    #[test]
    fn omelyan_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let leapfrog = GpuHmcLeapfrog::new(device, 16).unwrap();
        let omelyan = OmelyanIntegrator::new(leapfrog);
        assert!((omelyan.lambda() - OMELYAN_LAMBDA).abs() < 1e-15);
    }
}
