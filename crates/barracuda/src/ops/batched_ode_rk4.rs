// SPDX-License-Identifier: AGPL-3.0-or-later
//! `BatchedOdeRK4F64` — full-GPU parameter sweep for structured ODE systems.
//!
//! Extracted from `rk_stage.rs` (was D-S15-001 debt item): the single-trajectory
//! CPU-orchestrated `RkIntegrator` and the batched-GPU QS/c-di-GMP integrator
//! serve orthogonal purposes; separating them makes both easier to evolve.
//!
//! ## Current specialization
//!
//! The WGSL shader (`batched_qs_ode_rk4_f64.wgsl`) embeds the 5-variable
//! QS/c-di-GMP ODE derivative function directly.  This is an acknowledged
//! design compromise: WGSL has no function pointers, so generalization requires
//! either a shader-per-ODE or a DSL approach.  See debt item D-S15-001.
//!
//! ## State variables (5)
//! `[N, A, H, C, B]` = cell density, autoinducer, `HapR`, c-di-GMP, biofilm.
//!
//! ## Parameters per batch (17)
//! `[μ, K_cap, d_n, k_ai, d_ai, k_h, K_h, n_h, d_h, k_dgc, k_rep, k_pde, k_act,
//!   k_bio, K_bio, n_bio, d_bio]`

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Configuration for a batched RK4 parameter sweep.
#[derive(Debug, Clone)]
pub struct BatchedRk4Config {
    /// Number of independent parameter sets to integrate simultaneously.
    pub n_batches: u32,
    /// Number of fixed RK4 steps per trajectory.
    pub n_steps: u32,
    /// Fixed step size h (seconds or dimensionless time units).
    pub h: f64,
    /// Initial time t₀ (informational; the ODE is autonomous).
    pub t0: f64,
    /// Upper bound on any state variable (prevents blow-up).
    pub clamp_max: f64,
    /// Lower bound on any state variable (non-negativity).
    pub clamp_min: f64,
}

impl Default for BatchedRk4Config {
    fn default() -> Self {
        Self {
            n_batches: 1,
            n_steps: 1000,
            h: 0.01,
            t0: 0.0,
            clamp_max: 1e6,
            clamp_min: 0.0,
        }
    }
}

/// GPU uniform struct matching `QsOdeConfig` in the WGSL shader.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QsOdeConfigGpu {
    n_batches: u32,
    n_steps: u32,
    _pad0: u32,
    _pad1: u32,
    h: f64,
    t0: f64,
    clamp_max: f64,
    clamp_min: f64,
}

/// Batched RK4 integrator for the QS/c-di-GMP 5-variable ODE system.
///
/// Runs `B` independent parameter sets in parallel on the GPU.  Each GPU thread
/// integrates one complete trajectory using classic (fixed-step) RK4.
///
/// # Example
/// ```ignore
/// let cfg = BatchedRk4Config { n_batches: 10_000, n_steps: 500, h: 0.01, ..Default::default() };
/// let integrator = BatchedOdeRK4F64::new(device.clone(), cfg);
/// let finals = integrator.integrate(&initial_states, &batch_params)?;
/// ```
pub struct BatchedOdeRK4F64 {
    device: Arc<WgpuDevice>,
    config: BatchedRk4Config,
}

impl BatchedOdeRK4F64 {
    /// Number of state variables per trajectory.
    pub const N_VARS: usize = 5;
    /// Number of parameters per trajectory.
    pub const N_PARAMS: usize = 17;

    /// Create a new batched integrator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>, config: BatchedRk4Config) -> Self {
        Self { device, config }
    }

    /// Integrate all `n_batches` trajectories on the GPU.
    ///
    /// # Arguments
    /// * `initial_states` — flat `[B × 5]` f64 slice (row-major)
    /// * `batch_params`   — flat `[B × 17]` f64 slice (row-major)
    ///
    /// # Returns
    /// Flat `[B × 5]` f64 `Vec` with the final state for each batch.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn integrate(&self, initial_states: &[f64], batch_params: &[f64]) -> Result<Vec<f64>> {
        let b = self.config.n_batches as usize;

        if initial_states.len() != b * Self::N_VARS {
            return Err(BarracudaError::invalid_input(format!(
                "initial_states: expected {} ([B×5]), got {}",
                b * Self::N_VARS,
                initial_states.len()
            )));
        }
        if batch_params.len() != b * Self::N_PARAMS {
            return Err(BarracudaError::invalid_input(format!(
                "batch_params: expected {} ([B×17]), got {}",
                b * Self::N_PARAMS,
                batch_params.len()
            )));
        }

        let dev = &self.device;

        let cfg_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BatchedOdeRK4 Config"),
                contents: bytemuck::bytes_of(&QsOdeConfigGpu {
                    n_batches: self.config.n_batches,
                    n_steps: self.config.n_steps,
                    _pad0: 0,
                    _pad1: 0,
                    h: self.config.h,
                    t0: self.config.t0,
                    clamp_max: self.config.clamp_max,
                    clamp_min: self.config.clamp_min,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let init_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BatchedOdeRK4 InitStates"),
                contents: bytemuck::cast_slice(initial_states),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let param_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BatchedOdeRK4 Params"),
                contents: bytemuck::cast_slice(batch_params),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let out_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedOdeRK4 Output"),
            size: (b * Self::N_VARS * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        ComputeDispatch::new(dev, "BatchedOdeRK4")
            .shader(
                include_str!("../shaders/numerical/batched_qs_ode_rk4_f64.wgsl"),
                "main",
            )
            .f64()
            .uniform(0, &cfg_buf)
            .storage_read(1, &init_buf)
            .storage_read(2, &param_buf)
            .storage_rw(3, &out_buf)
            .dispatch_1d(b as u32)
            .submit()?;

        crate::utils::read_buffer_f64(dev, &out_buf, b * Self::N_VARS)
    }
}
