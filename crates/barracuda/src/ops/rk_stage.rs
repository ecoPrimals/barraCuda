// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runge-Kutta Stage Evaluation — CPU-Orchestrated RK4/RK45
//!
//! Single-trajectory ODE integration with Dormand-Prince adaptive stepping.
//!
//! **Use cases**:
//! - Soil dynamics and crop growth models (airSpring)
//! - Time-dependent nuclear physics ODEs (hotSpring)
//! - Population dynamics and ecosystem models (wetSpring)
//!
//! **Architecture**:
//! - CPU evaluates arbitrary `f(t, y)` closures (cannot be GPU-dispatched)
//! - For GPU-native structured ODEs, see [`Rk45AdaptiveGpu`](super::rk45_adaptive::Rk45AdaptiveGpu)
//! - For batched parameter sweeps, see [`BatchedRK4F64`](super::batched_rk4_sweep::BatchedRK4F64)
//!
//! **Deep Debt Principles**:
//! - Safe Rust (zero unsafe code)
//! - Hardware-agnostic (no vendor lock-in)

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// WGSL kernel for GPU-parallel multi-system RK4 integration (f32).
///
/// For f64 structured ODE batches, see [`BatchedOdeRK4F64`].
#[must_use]
pub fn wgsl_rk4_parallel() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!("../shaders/numerical/rk4_parallel_f64.wgsl").to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// RK45 (Dormand-Prince) coefficients
const DP_C: [f64; 6] = [0.0, 0.2, 0.3, 0.8, 8.0 / 9.0, 1.0];

const DP_B5: [f64; 6] = [
    35.0 / 384.0,
    0.0,
    500.0 / 1113.0,
    125.0 / 192.0,
    -2187.0 / 6784.0,
    11.0 / 84.0,
];

const DP_B4: [f64; 6] = [
    5179.0 / 57_600.0,
    0.0,
    7571.0 / 16_695.0,
    393.0 / 640.0,
    -92_097.0 / 339_200.0,
    187.0 / 2100.0,
];

/// Step-size control constants (Dormand-Prince adaptive stepping)
const STEP_SAFETY: f64 = 0.9;
const STEP_MAX_GROWTH: f64 = 2.0;
const STEP_GROW_EXPONENT: f64 = 0.2; // 1/p for RK5 (p=5)
const STEP_SHRINK_EXPONENT: f64 = 0.25; // 1/p for RK4 (p=4)

/// CPU-orchestrated RK45 integrator with GPU state update path.
///
/// The ODE right-hand-side `f(t, y)` is evaluated on CPU (arbitrary closures
/// cannot run on GPU). Linear-combination stage updates can be GPU-dispatched
/// for large state dimensions via [`Rk45AdaptiveGpu`](super::rk45_adaptive::Rk45AdaptiveGpu).
pub struct RkIntegrator {
    device: Arc<WgpuDevice>,
}

/// ODE function type: f(t, y) -> dy/dt
pub type OdeFunction = Box<dyn Fn(f64, &[f64]) -> Vec<f64> + Send + Sync>;

impl RkIntegrator {
    /// Create a new RK integrator backed by the given GPU device.
    /// The device is stored for future GPU-accelerated linear combination
    /// kernels (stage preparation, error estimation) when state dimension
    /// is large enough to amortize dispatch overhead.
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Access the underlying device (for callers that need it for batched dispatch).
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Integrate ODE system using RK45 (Dormand-Prince) with adaptive stepping
    /// # Arguments
    /// * `f` - ODE function: dy/dt = f(t, y)
    /// * `t0` - Initial time
    /// * `y0` - Initial state vector
    /// * `t_end` - Final time
    /// * `h_init` - Initial step size
    /// * `tol` - Error tolerance for adaptive stepping
    /// # Returns
    /// (times, states) - Vectors of time points and corresponding states
    /// # Errors
    /// Returns [`Err`] if state vector is empty.
    pub fn integrate_adaptive(
        &self,
        f: &OdeFunction,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h_init: f64,
        tol: f64,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let n = y0.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "State vector must have at least 1 dimension".to_string(),
            });
        }

        self.integrate_hybrid(f, t0, y0, t_end, h_init, tol)
    }

    /// Fixed-step RK4 integration (simpler, no error estimation)
    /// # Errors
    /// Returns [`Err`] if state vector is empty.
    pub fn integrate_fixed(
        &self,
        f: &OdeFunction,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h: f64,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        let n = y0.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "State vector must have at least 1 dimension".to_string(),
            });
        }

        let n_steps = ((t_end - t0) / h).ceil() as usize;
        let mut times = Vec::with_capacity(n_steps + 1);
        let mut states = Vec::with_capacity(n_steps + 1);

        let mut t = t0;
        let mut y = y0.to_vec();

        times.push(t);
        states.push(y.clone());

        for _ in 0..n_steps {
            let h_actual = (t_end - t).min(h);

            // RK4 stages
            let k1 = f(t, &y);
            let y1: Vec<f64> = y
                .iter()
                .zip(&k1)
                .map(|(yi, k)| yi + 0.5 * h_actual * k)
                .collect();

            let k2 = f(t + 0.5 * h_actual, &y1);
            let y2: Vec<f64> = y
                .iter()
                .zip(&k2)
                .map(|(yi, k)| yi + 0.5 * h_actual * k)
                .collect();

            let k3 = f(t + 0.5 * h_actual, &y2);
            let y3: Vec<f64> = y.iter().zip(&k3).map(|(yi, k)| yi + h_actual * k).collect();

            let k4 = f(t + h_actual, &y3);

            // Update
            for i in 0..n {
                y[i] += h_actual / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }
            t += h_actual;

            times.push(t);
            states.push(y.clone());

            if t >= t_end - 1e-10 {
                break;
            }
        }

        Ok((times, states))
    }

    /// CPU RK45 with adaptive stepping
    fn integrate_cpu(
        &self,
        f: &OdeFunction,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h_init: f64,
        tol: f64,
    ) -> (Vec<f64>, Vec<Vec<f64>>) {
        let n = y0.len();
        let mut times = vec![t0];
        let mut states = vec![y0.to_vec()];

        let mut t = t0;
        let mut y = y0.to_vec();
        let mut h = h_init;

        while t < t_end - 1e-10 {
            h = h.min(t_end - t);

            // RK45 Dormand-Prince stages
            let k1 = f(t, &y);

            let y1: Vec<f64> = (0..n).map(|i| y[i] + h * DP_C[1] * k1[i]).collect();
            let k2 = f(t + h * DP_C[1], &y1);

            let y2: Vec<f64> = (0..n)
                .map(|i| y[i] + h * (3.0 / 40.0 * k1[i] + 9.0 / 40.0 * k2[i]))
                .collect();
            let k3 = f(t + h * DP_C[2], &y2);

            let y3: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + h * (44.0 / 45.0 * k1[i] - 56.0 / 15.0 * k2[i] + 32.0 / 9.0 * k3[i])
                })
                .collect();
            let k4 = f(t + h * DP_C[3], &y3);

            let y4: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + h
                        * (19_372.0 / 6561.0 * k1[i] - 25_360.0 / 2187.0 * k2[i]
                            + 64_448.0 / 6561.0 * k3[i]
                            - 212.0 / 729.0 * k4[i])
                })
                .collect();
            let k5 = f(t + h * DP_C[4], &y4);

            let y5: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + h
                        * (9017.0 / 3168.0 * k1[i] - 355.0 / 33.0 * k2[i]
                            + 46_732.0 / 5247.0 * k3[i]
                            + 49.0 / 176.0 * k4[i]
                            - 5103.0 / 18_656.0 * k5[i])
                })
                .collect();
            let k6 = f(t + h * DP_C[5], &y5);

            // 5th order solution
            let y_new: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + h
                        * (DP_B5[0] * k1[i]
                            + DP_B5[2] * k3[i]
                            + DP_B5[3] * k4[i]
                            + DP_B5[4] * k5[i]
                            + DP_B5[5] * k6[i])
                })
                .collect();

            // 4th order solution for error estimate
            let y_4th: Vec<f64> = (0..n)
                .map(|i| {
                    y[i] + h
                        * (DP_B4[0] * k1[i]
                            + DP_B4[2] * k3[i]
                            + DP_B4[3] * k4[i]
                            + DP_B4[4] * k5[i]
                            + DP_B4[5] * k6[i])
                })
                .collect();

            // Error estimate
            let err: f64 = (0..n)
                .map(|i| (y_new[i] - y_4th[i]).powi(2))
                .sum::<f64>()
                .sqrt();

            if err < tol || h < 1e-12 {
                // Accept step
                t += h;
                y = y_new;
                times.push(t);
                states.push(y.clone());

                // Increase step size
                if err > 1e-15 {
                    h *= STEP_SAFETY * (tol / err).powf(STEP_GROW_EXPONENT);
                } else {
                    h *= STEP_MAX_GROWTH;
                }
            } else {
                // Reject step, decrease h
                h *= STEP_SAFETY * (tol / err).powf(STEP_SHRINK_EXPONENT);
            }

            h = h.min(t_end - t);
        }

        (times, states)
    }

    /// Hybrid GPU/CPU integration (GPU for stage prep, CPU for f evaluation)
    fn integrate_hybrid(
        &self,
        f: &OdeFunction,
        t0: f64,
        y0: &[f64],
        t_end: f64,
        h_init: f64,
        tol: f64,
    ) -> Result<(Vec<f64>, Vec<Vec<f64>>)> {
        // CPU adaptive RK45 — the error-controlled step rejection loop is
        // inherently serial (next step depends on error of current step).
        // GPU acceleration applies to the stage evaluations within each step,
        // which is handled by `BatchedOdeRK45F64` for large ODE systems.
        // 2. User-provided GPU f(t,y) kernel or CPU callback
        // 3. GPU error estimation kernel
        //
        // This is more complex due to the need for custom f(t,y) kernels
        // per problem. The CPU fallback is used for correctness.
        Ok(self.integrate_cpu(f, t0, y0, t_end, h_init, tol))
    }
}

// Re-export BatchedOdeRK4F64 from its dedicated module.
// See `batched_ode_rk4.rs` for implementation and documentation.
pub use super::batched_ode_rk4::{BatchedOdeRK4F64, BatchedRk4Config};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_sync;

    fn get_test_device() -> Arc<WgpuDevice> {
        get_test_device_sync()
    }

    #[test]
    fn test_exponential_decay() {
        let device = get_test_device();
        let integrator = RkIntegrator::new(device).unwrap();

        // dy/dt = -y, y(0) = 1 → y(t) = e^(-t)
        let f: OdeFunction = Box::new(|_t, y| vec![-y[0]]);

        let (times, states) = integrator
            .integrate_adaptive(&f, 0.0, &[1.0], 2.0, 0.1, 1e-6)
            .unwrap();

        let final_y = states.last().unwrap()[0];
        let expected = (-2.0_f64).exp();

        assert!(
            (final_y - expected).abs() < 1e-4,
            "y(2) = {final_y}, expected {expected}"
        );
        assert!(*times.last().unwrap() >= 2.0 - 1e-10);
    }

    #[test]
    fn test_harmonic_oscillator() {
        let device = get_test_device();
        let integrator = RkIntegrator::new(device).unwrap();

        // d²x/dt² = -x → x'' + x = 0
        // y = [x, x'] → y' = [x', -x]
        // x(0) = 1, x'(0) = 0 → x(t) = cos(t)
        let f: OdeFunction = Box::new(|_t, y| vec![y[1], -y[0]]);

        let (times, states) = integrator
            .integrate_fixed(&f, 0.0, &[1.0, 0.0], std::f64::consts::PI, 0.01)
            .unwrap();

        // At t = π, x should be -1
        let final_x = states.last().unwrap()[0];
        assert!(
            (final_x + 1.0).abs() < 0.01,
            "x(π) = {final_x}, expected -1"
        );

        // Check conservation: x² + (x')² = 1
        for state in &states {
            let energy = state[0].powi(2) + state[1].powi(2);
            assert!((energy - 1.0).abs() < 0.01, "Energy = {energy}, expected 1");
        }

        assert!(*times.last().unwrap() >= std::f64::consts::PI - 0.01);
    }

    #[test]
    fn test_lotka_volterra() {
        let device = get_test_device();
        let integrator = RkIntegrator::new(device).unwrap();

        // Predator-prey model
        // dx/dt = αx - βxy  (prey)
        // dy/dt = δxy - γy  (predator)
        let alpha = 1.1;
        let beta = 0.4;
        let delta = 0.1;
        let gamma = 0.4;

        let f: OdeFunction = Box::new(move |_t, y| {
            vec![
                alpha * y[0] - beta * y[0] * y[1],
                delta * y[0] * y[1] - gamma * y[1],
            ]
        });

        let (times, states) = integrator
            .integrate_adaptive(&f, 0.0, &[10.0, 10.0], 50.0, 0.1, 1e-4)
            .unwrap();

        // Check that populations remain positive
        for state in &states {
            assert!(state[0] > 0.0, "Prey went negative");
            assert!(state[1] > 0.0, "Predator went negative");
        }

        // Check that simulation completed
        assert!(*times.last().unwrap() >= 49.9);
    }
}
