// SPDX-License-Identifier: AGPL-3.0-or-later

//! Parallel Gillespie Stochastic Simulation Algorithm (SSA) — GPU f64.
//!
//! Runs N independent SSA trajectories in parallel, one GPU thread per
//! trajectory.  All trajectories are statistically independent; the PRNG
//! (xoshiro128**, inline in the shader) is seeded differently for each.
//!
//! ## Algorithm
//!
//! Direct method (Gillespie 1977) with mass-action propensities:
//!   `a_r = k_r × Π_s x_s! / (x_s - ν_r_s)!`  (`ν_r_s` = `stoich_reactant`[r,s])
//!
//! ## Limits
//!
//! - Reactions: unlimited (propensity scratch buffer [T × R] in storage; D-S21-002 resolved).
//! - Species: unlimited (storage buffers throughout).
//! - f64 throughout for species counts and times.
//!
//! ## Absorbed from
//!
//! wetSpring handoff §P1 Gillespie / SSA orchestration (Feb 2026).
//! Building blocks (`PrngXoshiro`, `SumReduceF64`) already present;
//! this shader composes them into a complete single-kernel SSA.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

// ─── GPU params struct (matches WGSL GillespieParams layout) ─────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GillespieParamsGpu {
    n_reactions: u32,
    n_species: u32,
    n_trajectories: u32,
    max_steps: u32,
    t_max: f64,
    _pad0: u32,
    _pad1: u32,
}

// ─── Public configuration ─────────────────────────────────────────────────────

/// Configuration for a Gillespie SSA run.
#[derive(Debug, Clone)]
pub struct GillespieConfig {
    /// Simulation end time.
    pub t_max: f64,
    /// Safety cap on iterations per trajectory (prevents infinite loops).
    pub max_steps: u32,
}

impl Default for GillespieConfig {
    fn default() -> Self {
        Self {
            t_max: 100.0,
            max_steps: 100_000,
        }
    }
}

/// Gillespie SSA result for all trajectories.
pub struct GillespieResult {
    /// Final species counts [T × S] (T trajectories, S species).
    pub states: Vec<f64>,
    /// Final simulation time for each trajectory [T].
    pub times: Vec<f64>,
    /// Number of trajectories T.
    pub n_trajectories: usize,
    /// Number of species S.
    pub n_species: usize,
}

// ─── Main operator ────────────────────────────────────────────────────────────

/// GPU-accelerated parallel Gillespie SSA (f64).
///
/// # Example
///
/// ```rust,ignore
/// # use barracuda::prelude::WgpuDevice;
/// # use barracuda::ops::bio::gillespie::{GillespieGpu, GillespieConfig};
/// # crate::device::test_pool::tokio_block_on(async {
/// let device = WgpuDevice::new().await.unwrap();
/// let ssa = GillespieGpu::new(&device);
///
/// // A → B with rate 1.0 (1 reaction, 2 species)
/// let rate_k       = vec![1.0_f64];
/// let stoich_react = vec![1u32, 0];  // reaction 0 consumes 1 A
/// let stoich_net   = vec![-1i32, 1]; // net: -1 A, +1 B
/// let initial_state = vec![100.0_f64, 0.0, 100.0, 0.0]; // 2 trajectories × 2 species
///
/// let seeds = vec![42u32, 0, 1, 0,  // trajectory 0 PRNG seed
///                  99u32, 0, 1, 0];  // trajectory 1 PRNG seed
///
/// let result = ssa.simulate(
///     &rate_k, &stoich_react, &stoich_net, &initial_state, &seeds, 2,
///     &GillespieConfig::default(),
/// ).unwrap();
/// # });
/// ```
pub struct GillespieGpu {
    device: Arc<WgpuDevice>,
}

impl GillespieGpu {
    /// Create Gillespie SSA simulator.
    #[must_use]
    pub fn new(device: &WgpuDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
        }
    }

    /// Run `n_trajectories` independent SSA trajectories in parallel.
    ///
    /// # Arguments
    /// - `rate_k`        : rate constants [R]
    /// - `stoich_react`  : reactant stoichiometry [R × S] (counts consumed)
    /// - `stoich_net`    : net stoichiometry [R × S] (change per firing)
    /// - `initial_states`: starting species counts [T × S]
    /// - `prng_seeds`    : xoshiro128** initial state [T × 4 u32]
    /// - `n_trajectories`: number of parallel trajectories T
    /// - `config`        : simulation parameters
    ///
    /// # Panics
    /// Panics if `initial_states.len() != n_trajectories * n_species` or `prng_seeds.len() != n_trajectories * 4`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    #[expect(clippy::too_many_arguments, reason = "API")]
    pub fn simulate(
        &self,
        rate_k: &[f64],
        stoich_react: &[u32],
        stoich_net: &[i32],
        initial_states: &[f64],
        prng_seeds: &[u32],
        n_trajectories: usize,
        config: &GillespieConfig,
    ) -> Result<GillespieResult> {
        let dev = &self.device;
        let n_r = rate_k.len();
        let n_s = stoich_net.len() / n_r;
        let n_t = n_trajectories;

        assert_eq!(
            initial_states.len(),
            n_t * n_s,
            "initial_states must be [T×S]"
        );
        assert_eq!(prng_seeds.len(), n_t * 4, "prng_seeds must be [T×4]");

        // ── Upload read-only buffers ───────────────────────────────────────────
        let rate_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gillespie rates"),
                contents: bytemuck::cast_slice(rate_k),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let sreact_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gillespie stoich_react"),
                contents: bytemuck::cast_slice(stoich_react),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let snet_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gillespie stoich_net"),
                contents: bytemuck::cast_slice(stoich_net),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // ── Mutable buffers (states, prng, times) ─────────────────────────────
        let states_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gillespie states"),
                contents: bytemuck::cast_slice(initial_states),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let prng_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gillespie prng"),
                contents: bytemuck::cast_slice(prng_seeds),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let times_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gillespie times"),
            size: (n_t * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Per-thread propensity scratch: [T × R] f64 (replaces static array<f64,32>)
        let prop_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gillespie propensities"),
            size: ((n_t * n_r) * std::mem::size_of::<f64>()).max(8) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let params = GillespieParamsGpu {
            n_reactions: n_r as u32,
            n_species: n_s as u32,
            n_trajectories: n_t as u32,
            max_steps: config.max_steps,
            t_max: config.t_max,
            _pad0: 0,
            _pad1: 0,
        };
        // Storage (not uniform) because GillespieParamsGpu contains f64 (t_max).
        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Gillespie params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::STORAGE,
            });

        // ── Dispatch (f64-aware: exp/log patching + ILP optimizer) ─────────────
        let wg = (n_t as u32).div_ceil(WORKGROUP_SIZE_1D);
        ComputeDispatch::new(dev, "gillespie_ssa")
            .shader(
                include_str!("../../shaders/bio/gillespie_ssa_f64.wgsl"),
                "main",
            )
            .f64()
            .storage_read(0, &params_buf)
            .storage_read(1, &rate_buf)
            .storage_read(2, &sreact_buf)
            .storage_read(3, &snet_buf)
            .storage_rw(4, &states_buf)
            .storage_rw(5, &prng_buf)
            .storage_rw(6, &times_buf)
            .storage_rw(7, &prop_buf)
            .dispatch(wg, 1, 1)
            .submit()?;

        // ── Read back results ─────────────────────────────────────────────────
        let states = dev.read_buffer_f64(&states_buf, n_t * n_s)?;
        let times = dev.read_buffer_f64(&times_buf, n_t)?;

        Ok(GillespieResult {
            states,
            times,
            n_trajectories: n_t,
            n_species: n_s,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    #[tokio::test]
    async fn test_irreversible_decay_mean() {
        // A → ∅  with k=1.0, x0=100, t_max=1.0
        // Mean: E[X(t)] = 100 × e^{-1} ≈ 36.8
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let ssa = GillespieGpu::new(&device);

        let n_traj = 256usize;
        let rate_k = vec![1.0_f64]; // k = 1
        let stoich_react = vec![1u32]; // consumes 1 A
        let stoich_net = vec![-1i32]; // net: -1

        // Initial state: 100 for each trajectory
        let initial: Vec<f64> = vec![100.0; n_traj];

        // Seeds: different for each trajectory
        let mut seeds = Vec::with_capacity(n_traj * 4);
        for i in 0..n_traj {
            seeds.push(i as u32 + 1);
            seeds.push(0x9e3779b9u32);
            seeds.push(0x6c62272eu32);
            seeds.push(0x85ebca77u32);
        }

        let result = ssa
            .simulate(
                &rate_k,
                &stoich_react,
                &stoich_net,
                &initial,
                &seeds,
                n_traj,
                &GillespieConfig {
                    t_max: 1.0,
                    max_steps: 100_000,
                },
            )
            .unwrap();

        let mean: f64 = result.states.iter().sum::<f64>() / n_traj as f64;
        let expected = 100.0 * (-1.0_f64).exp();
        // Allow 10% relative tolerance (stochastic)
        assert!(
            (mean - expected).abs() / expected < 0.15,
            "mean={mean:.2}, expected≈{expected:.2}"
        );
    }

    #[tokio::test]
    async fn test_absorbing_state() {
        // System with zero rate → all trajectories stay at initial
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let ssa = GillespieGpu::new(&device);

        let rate_k = vec![0.0_f64];
        let stoich_react = vec![1u32];
        let stoich_net = vec![-1i32];
        let initial = vec![50.0_f64, 50.0];
        let seeds = vec![1u32, 2, 3, 4, 5, 6, 7, 8];

        let result = ssa
            .simulate(
                &rate_k,
                &stoich_react,
                &stoich_net,
                &initial,
                &seeds,
                2,
                &GillespieConfig::default(),
            )
            .unwrap();

        assert_eq!(result.states[0], 50.0);
        assert_eq!(result.states[1], 50.0);
    }
}
