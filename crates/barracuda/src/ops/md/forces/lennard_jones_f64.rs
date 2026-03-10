// SPDX-License-Identifier: AGPL-3.0-only
//! Lennard-Jones Force (f64) — Science-Grade MD
//!
//! **Deep Debt Evolution (Feb 16, 2026)**:
//! - ✅ Pure WGSL f64 implementation
//! - ✅ Native sqrt(f64) via Vulkan
//! - ✅ Hardware-agnostic (NVIDIA/AMD/Intel via WebGPU)
//! - ✅ WGSL as unified math language
//!
//! **Physics**: Van der Waals interactions (noble gases, simple liquids)
//! **Potential**: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
//! **Force**: F = 24ε/r * [2(σ/r)^12 - (σ/r)^6] * r̂

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

const WGSL_DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
const WGSL_DF64_TRANSCENDENTALS: &str =
    include_str!("../../../shaders/math/df64_transcendentals.wgsl");
const LJ_SHADER_DF64: &str = include_str!("lennard_jones_df64.wgsl");

/// f64 Lennard-Jones force computation
///
/// **Key features**:
/// - Full f64 precision for energy conservation
/// - Lorentz-Berthelot mixing rules for heterogeneous systems
/// - Optional shifted potential for smooth cutoff
/// - Per-particle or global parameters
pub struct LennardJonesF64;

impl LennardJonesF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("lennard_jones_f64.wgsl")
    }

    fn wgsl_shader_for_device(device: &WgpuDevice) -> String {
        let profile = GpuDriverProfile::from_device(device);
        let strategy = profile.fp64_strategy();
        tracing::info!(?strategy, "LJ F64: using {:?} FP64 strategy", strategy);
        match strategy {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                Self::wgsl_shader().to_string()
            }
            Fp64Strategy::Hybrid => {
                format!("{WGSL_DF64_CORE}\n{WGSL_DF64_TRANSCENDENTALS}\n{LJ_SHADER_DF64}")
            }
        }
    }

    /// Compute LJ forces with per-particle parameters
    ///
    /// # Arguments
    /// * `device` - GPU device
    /// * `positions` - Particle positions [N*3] (x,y,z interleaved)
    /// * `sigmas` - Per-particle σ values [N]
    /// * `epsilons` - Per-particle ε values [N]
    /// * `cutoff` - Cutoff radius
    ///
    /// # Returns
    /// Forces [N*3] (fx,fy,fz interleaved)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(
        device: Arc<WgpuDevice>,
        positions: &[f64],
        sigmas: &[f64],
        epsilons: &[f64],
        cutoff: f64,
    ) -> Result<Vec<f64>> {
        let n = sigmas.len();
        if positions.len() != n * 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n * 3],
                actual: vec![positions.len()],
            });
        }
        if epsilons.len() != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: vec![epsilons.len()],
            });
        }

        // Create buffers
        let pos_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LJ F64 Positions"),
                contents: bytemuck::cast_slice(positions),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let sigma_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LJ F64 Sigmas"),
                contents: bytemuck::cast_slice(sigmas),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let epsilon_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LJ F64 Epsilons"),
                contents: bytemuck::cast_slice(epsilons),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let forces_buffer = device.create_buffer_f64(n * 3)?;

        // Params: n_particles (u32), _pad (u32), cutoff (f64), cutoff_sq (f64)
        let cutoff_sq = cutoff * cutoff;
        let mut params_bytes = Vec::with_capacity(24);
        params_bytes.extend_from_slice(&(n as u32).to_le_bytes());
        params_bytes.extend_from_slice(&0u32.to_le_bytes()); // padding
        params_bytes.extend_from_slice(&cutoff.to_le_bytes());
        params_bytes.extend_from_slice(&cutoff_sq.to_le_bytes());

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LJ F64 Params"),
                contents: &params_bytes,
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let src = Self::wgsl_shader_for_device(&device);
        ComputeDispatch::new(&device, "LJ F64")
            .shader(&src, "lennard_jones_f64")
            .f64()
            .storage_read(0, &pos_buffer)
            .storage_read(1, &sigma_buffer)
            .storage_read(2, &epsilon_buffer)
            .storage_rw(3, &forces_buffer)
            .uniform(4, &params_buffer)
            .dispatch((n as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1)
            .submit()?;

        crate::utils::read_buffer_f64(&device, &forces_buffer, n * 3)
    }

    /// Compute LJ forces with uniform parameters (all particles same σ, ε)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute_uniform(
        device: Arc<WgpuDevice>,
        positions: &[f64],
        sigma: f64,
        epsilon: f64,
        cutoff: f64,
    ) -> Result<Vec<f64>> {
        let n = positions.len() / 3;
        let sigmas = vec![sigma; n];
        let epsilons = vec![epsilon; n];
        Self::compute(device, positions, &sigmas, &epsilons, cutoff)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_lj_f64_two_particles() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        if !device
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            println!("Skipping: GPU does not support SHADER_F64");
            return;
        }

        // Two argon-like particles at distance 3.4 Å (σ = 3.4, ε = 1.0)
        // At r = σ, U = 0, F is repulsive
        let positions: Vec<f64> = vec![0.0, 0.0, 0.0, 3.4, 0.0, 0.0];
        let sigma = 3.4;
        let epsilon = 1.0;
        let cutoff = 10.0;

        let forces =
            LennardJonesF64::compute_uniform(device, &positions, sigma, epsilon, cutoff).unwrap();

        assert_eq!(forces.len(), 6);

        // At r = σ, the force should be repulsive (pushing particle 0 to -x, particle 1 to +x)
        // Force magnitude: F = 24ε/r * [2(σ/r)^12 - (σ/r)^6] = 24/3.4 * [2 - 1] = 24/3.4 ≈ 7.06
        // Particle 0 should have negative fx (pushed away from particle 1)
        assert!(
            forces[0] < 0.0,
            "Particle 0 should be pushed in -x direction, got fx={}",
            forces[0]
        );

        // Particle 1 should have positive fx (pushed away from particle 0)
        assert!(
            forces[3] > 0.0,
            "Particle 1 should be pushed in +x direction, got fx={}",
            forces[3]
        );

        // Newton's third law: forces should be equal and opposite
        assert!(
            (forces[0] + forces[3]).abs() < 1e-10,
            "Newton's third law violated: {} + {} = {}",
            forces[0],
            forces[3],
            forces[0] + forces[3]
        );

        println!("✅ LJ f64 two-particle test passed");
    }

    #[tokio::test]
    async fn test_lj_f64_equilibrium() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        if !device
            .device
            .features()
            .contains(wgpu::Features::SHADER_F64)
        {
            return;
        }

        // At r = 2^(1/6) * σ ≈ 1.122 * σ, force should be zero (equilibrium)
        let sigma = 1.0;
        let r_eq = sigma * 2.0_f64.powf(1.0 / 6.0);
        let positions: Vec<f64> = vec![0.0, 0.0, 0.0, r_eq, 0.0, 0.0];

        let forces =
            LennardJonesF64::compute_uniform(device, &positions, sigma, 1.0, 10.0).unwrap();

        // Force should be very small at equilibrium
        assert!(
            forces[0].abs() < 1e-10,
            "Force should be ~0 at equilibrium, got {}",
            forces[0]
        );

        println!("✅ LJ f64 equilibrium test passed");
    }
}
