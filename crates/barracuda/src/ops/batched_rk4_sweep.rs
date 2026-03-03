// SPDX-License-Identifier: AGPL-3.0-only

//! `BatchedRK4F64` — General-purpose N-trajectory RK4/RK45 orchestration.
//!
//! Wraps [`RkIntegrator`] to run N independent ODE instances in parallel,
//! each with its own `f(t, y) → dy/dt` closure (parameters baked in via closure).
//!
//! ## When to use
//! - Parameter sweeps: N ODE instances with different rate constants
//! - Monte Carlo: N independent trajectories from different initial conditions
//! - Ensemble integration: uncertainty quantification over N parameter draws
//!
//! ## When NOT to use
//! - Fixed structured ODE across batches → `BatchedOdeRK4F64` (full-GPU, zero CPU callback)
//!
//! ## Design
//! - CPU orchestration: `std::thread::scope` — one thread per instance
//! - GPU state updates: each `RkIntegrator` dispatches GPU compute for D ≥ 128
//! - For D < 128 each integrator falls back to pure-CPU (no device saturation needed)
//! - Zero `unsafe` code; thread safety via `Fn + Send + Sync` bound on closures
//!
//! Resolves D-S21-001 from DEBT.md (Feb 2026 wetSpring handoff).

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::ops::rk_stage::RkIntegrator;
use std::sync::Arc;

/// Type alias for ODE right-hand-side functions: `f(t, y) -> dy/dt`.
///
/// Must be `Send + Sync` for parallel trajectory integration.
pub type OdeFn = Box<dyn Fn(f64, &[f64]) -> Vec<f64> + Send + Sync>;

// ─── Result type ─────────────────────────────────────────────────────────────

/// Integration result for one trajectory in a batched run.
#[derive(Debug, Clone)]
pub struct TrajectoryResult {
    /// Time points at which state was recorded.
    pub times: Vec<f64>,
    /// State snapshots; `states[i]` corresponds to `times[i]`.
    pub states: Vec<Vec<f64>>,
    /// Zero-based index of this instance within the batch.
    pub instance: usize,
}

impl TrajectoryResult {
    /// Final state vector (last element of `states`).
    pub fn final_state(&self) -> &[f64] {
        self.states.last().map(Vec::as_slice).unwrap_or_default()
    }

    /// Final time reached.
    pub fn final_time(&self) -> f64 {
        self.times.last().copied().unwrap_or(0.0)
    }
}

// ─── Main orchestrator ───────────────────────────────────────────────────────

/// N-trajectory RK4/RK45 orchestrator.
///
/// # Example — three exponential decays with different rates
///
/// ```rust,ignore
/// # use barracuda::prelude::WgpuDevice;
/// # use barracuda::ops::batched_rk4_sweep::BatchedRK4F64;
/// # crate::device::test_pool::tokio_block_on(async {
/// let device = WgpuDevice::new().await.unwrap();
/// let batcher = BatchedRK4F64::new(&device);
///
/// let rates = [0.5_f64, 1.0, 2.0];
/// let odes: Vec<Box<dyn Fn(f64, &[f64]) -> Vec<f64> + Send + Sync>> = rates
///     .iter()
///     .map(|&k| -> Box<dyn Fn(f64, &[f64]) -> Vec<f64> + Send + Sync> {
///         Box::new(move |_t, y| vec![-k * y[0]])
///     })
///     .collect();
///
/// let y0_batch = vec![vec![1.0_f64]; 3];
/// let results = batcher.integrate_fixed(&odes, 0.0, &y0_batch, 5.0, 0.01).unwrap();
/// for r in &results {
///     println!("instance {}: final y = {:.4}", r.instance, r.final_state()[0]);
/// }
/// # });
/// ```
pub struct BatchedRK4F64 {
    device: Arc<WgpuDevice>,
}

impl BatchedRK4F64 {
    /// Create a new batcher bound to `device`.
    pub fn new(device: &WgpuDevice) -> Self {
        Self {
            device: Arc::new(device.clone()),
        }
    }

    // ─── Fixed-step RK4 ──────────────────────────────────────────────────────

    /// Run N ODE instances in parallel with fixed-step RK4.
    ///
    /// # Arguments
    /// - `odes`     : one closure per instance; each captures its own parameters
    /// - `t0`       : start time (shared across instances)
    /// - `y0_batch` : initial conditions `[N][D]`; all must have the same D
    /// - `t_end`    : end time (shared)
    /// - `h`        : fixed step size (shared)
    ///
    /// # Returns
    /// `Vec<TrajectoryResult>` in the same order as `odes`.
    pub fn integrate_fixed(
        &self,
        odes: &[OdeFn],
        t0: f64,
        y0_batch: &[Vec<f64>],
        t_end: f64,
        h: f64,
    ) -> Result<Vec<TrajectoryResult>> {
        self.validate_batch(odes.len(), y0_batch)?;
        if odes.is_empty() {
            return Ok(vec![]);
        }

        let device = &self.device;
        let mut results: Vec<Option<Result<TrajectoryResult>>> =
            (0..odes.len()).map(|_| None).collect();

        std::thread::scope(|scope| {
            let handles: Vec<_> = odes
                .iter()
                .zip(y0_batch.iter())
                .enumerate()
                .map(|(idx, (f, y0))| {
                    let dev = device.clone();
                    scope.spawn(move || -> Result<TrajectoryResult> {
                        let integrator = RkIntegrator::new(dev)?;
                        // `f: &Box<dyn Fn...>` = `&OdeFunction`; pass directly.
                        let (times, states) = integrator.integrate_fixed(f, t0, y0, t_end, h)?;
                        Ok(TrajectoryResult {
                            times,
                            states,
                            instance: idx,
                        })
                    })
                })
                .collect();

            for (i, handle) in handles.into_iter().enumerate() {
                results[i] = Some(handle.join().unwrap_or_else(|_| {
                    Err(BarracudaError::Internal(format!(
                        "BatchedRK4F64::integrate_fixed: thread panic for instance {i}"
                    )))
                }));
            }
        });

        results
            .into_iter()
            .map(|r| r.expect("join filled every slot"))
            .collect()
    }

    // ─── Adaptive RK45 ───────────────────────────────────────────────────────

    /// Run N ODE instances in parallel with adaptive-step RK45 (Dormand-Prince).
    ///
    /// # Arguments
    /// - `odes`     : one closure per instance
    /// - `t0`       : start time
    /// - `y0_batch` : initial conditions `[N][D]`
    /// - `t_end`    : end time
    /// - `h_init`   : initial step size
    /// - `tol`      : local error tolerance for adaptive stepping
    pub fn integrate_adaptive(
        &self,
        odes: &[OdeFn],
        t0: f64,
        y0_batch: &[Vec<f64>],
        t_end: f64,
        h_init: f64,
        tol: f64,
    ) -> Result<Vec<TrajectoryResult>> {
        self.validate_batch(odes.len(), y0_batch)?;
        if odes.is_empty() {
            return Ok(vec![]);
        }

        let device = &self.device;
        let mut results: Vec<Option<Result<TrajectoryResult>>> =
            (0..odes.len()).map(|_| None).collect();

        std::thread::scope(|scope| {
            let handles: Vec<_> = odes
                .iter()
                .zip(y0_batch.iter())
                .enumerate()
                .map(|(idx, (f, y0))| {
                    let dev = device.clone();
                    scope.spawn(move || -> Result<TrajectoryResult> {
                        let integrator = RkIntegrator::new(dev)?;
                        let (times, states) =
                            integrator.integrate_adaptive(f, t0, y0, t_end, h_init, tol)?;
                        Ok(TrajectoryResult {
                            times,
                            states,
                            instance: idx,
                        })
                    })
                })
                .collect();

            for (i, handle) in handles.into_iter().enumerate() {
                results[i] = Some(handle.join().unwrap_or_else(|_| {
                    Err(BarracudaError::Internal(format!(
                        "BatchedRK4F64::integrate_adaptive: thread panic for instance {i}"
                    )))
                }));
            }
        });

        results
            .into_iter()
            .map(|r| r.expect("join filled every slot"))
            .collect()
    }

    // ─── Private helpers ─────────────────────────────────────────────────────

    fn validate_batch(&self, n_odes: usize, y0_batch: &[Vec<f64>]) -> Result<()> {
        if y0_batch.len() != n_odes {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "BatchedRK4F64: odes.len()={n_odes} but y0_batch.len()={}",
                    y0_batch.len()
                ),
            });
        }
        if n_odes == 0 {
            return Ok(());
        }
        let d = y0_batch[0].len();
        for (i, y) in y0_batch.iter().enumerate() {
            if y.len() != d {
                return Err(BarracudaError::InvalidInput {
                    message: format!(
                        "BatchedRK4F64: instance {i} has D={} but expected D={d}",
                        y.len()
                    ),
                });
            }
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    /// Exponential decay: dy/dt = -k*y  →  y(t) = y0 * exp(-k*t)
    fn decay_ode(k: f64) -> OdeFn {
        Box::new(move |_t, y| vec![-k * y[0]])
    }

    #[tokio::test]
    async fn test_fixed_three_decay_rates() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let batcher = BatchedRK4F64::new(&device);

        let rates = [0.5_f64, 1.0, 2.0];
        let odes: Vec<_> = rates.iter().map(|&k| decay_ode(k)).collect();
        let y0_batch: Vec<_> = (0..3).map(|_| vec![1.0_f64]).collect();

        let results = batcher
            .integrate_fixed(&odes, 0.0, &y0_batch, 5.0, 0.01)
            .unwrap();

        assert_eq!(results.len(), 3);
        for (r, &k) in results.iter().zip(&rates) {
            let y_final = r.final_state()[0];
            let y_exact = (-k * 5.0_f64).exp();
            let rel_error = (y_final - y_exact).abs() / y_exact;
            assert!(
                rel_error < 1e-4,
                "instance {}: y_final={y_final:.6}, y_exact={y_exact:.6}, rel_err={rel_error:.2e}",
                r.instance
            );
        }
    }

    #[tokio::test]
    async fn test_adaptive_single_instance() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let batcher = BatchedRK4F64::new(&device);

        let odes: Vec<_> = vec![decay_ode(1.0)];
        let y0_batch = vec![vec![1.0_f64]];

        let results = batcher
            .integrate_adaptive(&odes, 0.0, &y0_batch, 3.0, 0.1, 1e-6)
            .unwrap();

        let y_final = results[0].final_state()[0];
        let y_exact = (-3.0_f64).exp();
        let rel_error = (y_final - y_exact).abs() / y_exact;
        assert!(
            rel_error < 1e-4,
            "y_final={y_final:.6}, y_exact={y_exact:.6}"
        );
    }

    #[tokio::test]
    async fn test_empty_batch() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let batcher = BatchedRK4F64::new(&device);
        let results = batcher.integrate_fixed(&[], 0.0, &[], 1.0, 0.01).unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_dimension_mismatch_error() {
        let Some(device) = test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let batcher = BatchedRK4F64::new(&device);
        let odes: Vec<_> = vec![decay_ode(1.0), decay_ode(2.0)];
        // y0_batch has wrong inner dimension for instance 1
        let y0_batch = vec![vec![1.0_f64], vec![1.0_f64, 0.0_f64]];
        assert!(batcher
            .integrate_fixed(&odes, 0.0, &y0_batch, 1.0, 0.01)
            .is_err());
    }
}
