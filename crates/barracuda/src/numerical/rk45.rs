// SPDX-License-Identifier: AGPL-3.0-only
//! Adaptive Runge-Kutta-Fehlberg (RK45) ODE Solver
//!
//! Implements the Dormand-Prince (DOPRI5) embedded method for solving
//! initial value problems with automatic step size control.
//!
//! # Algorithm
//!
//! Uses a 5th-order Runge-Kutta method with an embedded 4th-order estimate
//! for local error control. Step sizes are adjusted to maintain the local
//! error below a specified tolerance.
//!
//! # System Form
//!
//! Solves: dy/dt = f(t, y)
//!
//! with initial condition y(t₀) = y₀.
//!
//! # Applications
//!
//! - **Chemical kinetics**: Stiff reaction systems
//! - **Orbital mechanics**: N-body problems
//! - **Population dynamics**: Lotka-Volterra equations
//! - **Quantum dynamics**: Time-dependent Schrödinger
//!
//! # References
//!
//! - Dormand, J. R. & Prince, P. J. (1980)
//! - Numerical Recipes, §17.2

/// WGSL kernel for Dormand-Prince RK45 adaptive ODE stepping (f64).
pub const WGSL_RK45_F64: &str = include_str!("../shaders/math/rk45_f64.wgsl");

use crate::error::{BarracudaError, Result};

/// Configuration for the RK45 solver.
#[derive(Debug, Clone)]
pub struct Rk45Config {
    /// Relative tolerance
    pub rtol: f64,
    /// Absolute tolerance
    pub atol: f64,
    /// Initial step size
    pub h_init: f64,
    /// Minimum step size
    pub h_min: f64,
    /// Maximum step size
    pub h_max: f64,
    /// Maximum number of steps
    pub max_steps: usize,
    /// Safety factor for step size adjustment (0.8-0.9 typical)
    pub safety: f64,
}

impl Default for Rk45Config {
    fn default() -> Self {
        Self {
            rtol: 1e-6,
            atol: 1e-9,
            h_init: 0.01,
            h_min: 1e-12,
            h_max: 1.0,
            max_steps: 100_000,
            safety: 0.9,
        }
    }
}

impl Rk45Config {
    /// Create a new configuration with specified tolerances.
    #[must_use]
    pub fn new(rtol: f64, atol: f64) -> Self {
        Self {
            rtol,
            atol,
            ..Default::default()
        }
    }

    /// Set the initial step size.
    #[must_use]
    pub fn with_h_init(mut self, h: f64) -> Self {
        self.h_init = h;
        self
    }

    /// Set step size bounds.
    #[must_use]
    pub fn with_step_bounds(mut self, h_min: f64, h_max: f64) -> Self {
        self.h_min = h_min;
        self.h_max = h_max;
        self
    }

    /// Set maximum number of steps.
    #[must_use]
    pub fn with_max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Set safety factor for step size adjustment.
    #[must_use]
    pub fn with_safety(mut self, safety: f64) -> Self {
        self.safety = safety;
        self
    }
}

/// Result of the RK45 integration.
#[derive(Debug)]
pub struct Rk45Result {
    /// Final time reached
    pub t_final: f64,
    /// Solution at final time
    pub y_final: Vec<f64>,
    /// Time points (if history was recorded)
    pub t_history: Vec<f64>,
    /// Solution history (if recorded)
    pub y_history: Vec<Vec<f64>>,
    /// Number of successful steps
    pub n_steps: usize,
    /// Number of rejected steps
    pub n_rejected: usize,
    /// Final step size
    pub h_final: f64,
}

impl Rk45Result {
    /// Number of variables in the ODE system.
    ///
    /// Derived from `y_final.len()`. Returns 0 if `y_final` is empty.
    #[must_use]
    pub fn n_vars(&self) -> usize {
        self.y_final.len()
    }

    /// Extract the trajectory of a single variable across all time steps.
    ///
    /// For an ODE system with `n_vars` variables, this collects
    /// `y_history[step][var_idx]` for every recorded step into a single
    /// `Vec<f64>`, eliminating the manual flat-array indexing pattern
    /// used by ODE scenario builders.
    ///
    /// Returns an empty `Vec` if `var_idx >= n_vars` or no history was recorded.
    #[must_use]
    pub fn variable_trajectory(&self, var_idx: usize) -> Vec<f64> {
        self.y_history
            .iter()
            .filter_map(|step| step.get(var_idx).copied())
            .collect()
    }
}

/// Adaptive RK45 ODE solver.
///
/// Solves the system dy/dt = f(t, y) from `t_start` to `t_end`.
///
/// # Arguments
///
/// * `f` - The derivative function f(t, y) -> dy/dt
/// * `t_start` - Initial time
/// * `t_end` - Final time
/// * `y0` - Initial conditions
/// * `config` - Solver configuration
///
/// # Returns
///
/// [`Rk45Result`] containing the solution and diagnostics.
///
/// # Errors
///
/// Returns [`Err`] if `t_end <= t_start`, `y0` is empty, step size falls below
/// minimum, or max steps exceeded.
///
/// # Example
///
/// ```
/// use barracuda::numerical::rk45::{rk45_solve, Rk45Config};
///
/// // Solve dy/dt = -y (exponential decay)
/// let f = |_t: f64, y: &[f64]| vec![-y[0]];
/// let config = Rk45Config::new(1e-6, 1e-9);
///
/// let result = rk45_solve(&f, 0.0, 1.0, &[1.0], &config).unwrap();
///
/// // y(1) ≈ e^(-1) ≈ 0.368
/// assert!((result.y_final[0] - (-1.0_f64).exp()).abs() < 1e-5);
/// ```
pub fn rk45_solve<F>(
    f: &F,
    t_start: f64,
    t_end: f64,
    y0: &[f64],
    config: &Rk45Config,
) -> Result<Rk45Result>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    if t_end <= t_start {
        return Err(BarracudaError::InvalidInput {
            message: "t_end must be > t_start".to_string(),
        });
    }

    if y0.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "y0 must not be empty".to_string(),
        });
    }

    let n = y0.len();
    let mut t = t_start;
    let mut y = y0.to_vec();
    let mut h = config.h_init.min(t_end - t_start);

    let mut t_history = vec![t];
    let mut y_history = vec![y.clone()];
    let mut n_steps = 0;
    let mut n_rejected = 0;

    // Dormand-Prince coefficients
    const A21: f64 = 1.0 / 5.0;
    const A31: f64 = 3.0 / 40.0;
    const A32: f64 = 9.0 / 40.0;
    const A41: f64 = 44.0 / 45.0;
    const A42: f64 = -56.0 / 15.0;
    const A43: f64 = 32.0 / 9.0;
    const A51: f64 = 19372.0 / 6561.0;
    const A52: f64 = -25360.0 / 2187.0;
    const A53: f64 = 64448.0 / 6561.0;
    const A54: f64 = -212.0 / 729.0;
    const A61: f64 = 9017.0 / 3168.0;
    const A62: f64 = -355.0 / 33.0;
    const A63: f64 = 46732.0 / 5247.0;
    const A64: f64 = 49.0 / 176.0;
    const A65: f64 = -5103.0 / 18656.0;
    const A71: f64 = 35.0 / 384.0;
    const A73: f64 = 500.0 / 1113.0;
    const A74: f64 = 125.0 / 192.0;
    const A75: f64 = -2187.0 / 6784.0;
    const A76: f64 = 11.0 / 84.0;

    // Error estimation coefficients (5th order - 4th order)
    const E1: f64 = 71.0 / 57_600.0;
    const E3: f64 = -71.0 / 16_695.0;
    const E4: f64 = 71.0 / 1920.0;
    const E5: f64 = -17_253.0 / 339_200.0;
    const E6: f64 = 22.0 / 525.0;
    const E7: f64 = -1.0 / 40.0;

    while t < t_end && n_steps < config.max_steps {
        // Don't step past t_end
        if t + h > t_end {
            h = t_end - t;
        }

        // Compute RK stages
        let k1 = f(t, &y);

        let y2: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .map(|(yi, k1i)| yi + h * A21 * k1i)
            .collect();
        let k2 = f(t + h / 5.0, &y2);

        let y3: Vec<f64> = y
            .iter()
            .zip(k1.iter())
            .zip(k2.iter())
            .map(|((yi, k1i), k2i)| yi + h * (A31 * k1i + A32 * k2i))
            .collect();
        let k3 = f(t + 3.0 * h / 10.0, &y3);

        let y4: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]))
            .collect();
        let k4 = f(t + 4.0 * h / 5.0, &y4);

        let y5: Vec<f64> = (0..n)
            .map(|i| y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]))
            .collect();
        let k5 = f(t + 8.0 * h / 9.0, &y5);

        let y6: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i])
            })
            .collect();
        let k6 = f(t + h, &y6);

        // 5th order solution (y_new)
        let y_new: Vec<f64> = (0..n)
            .map(|i| {
                y[i] + h * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i])
            })
            .collect();

        // Compute k7 for error estimate
        let k7 = f(t + h, &y_new);

        // Error estimate
        let error: Vec<f64> = (0..n)
            .map(|i| {
                h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i])
            })
            .collect();

        // Compute error norm
        let err_norm = error
            .iter()
            .zip(y_new.iter())
            .map(|(e, yi)| {
                let scale = config.atol + config.rtol * yi.abs();
                (e / scale).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / (n as f64).sqrt();

        if err_norm <= 1.0 {
            // Accept step
            t += h;
            y = y_new;
            t_history.push(t);
            y_history.push(y.clone());
            n_steps += 1;

            // Compute new step size
            if err_norm > 0.0 {
                let h_new = h * config.safety * (1.0 / err_norm).powf(0.2);
                h = h_new.clamp(config.h_min, config.h_max);
            } else {
                h = config.h_max;
            }
        } else {
            // Reject step, reduce h
            let h_new = h * config.safety * (1.0 / err_norm).powf(0.25);
            h = h_new.max(config.h_min);
            n_rejected += 1;

            if h <= config.h_min {
                return Err(BarracudaError::Numerical {
                    message: format!("Step size {} below minimum {} at t={}", h, config.h_min, t),
                });
            }
        }
    }

    if n_steps >= config.max_steps {
        return Err(BarracudaError::Numerical {
            message: format!("Max steps {} exceeded", config.max_steps),
        });
    }

    Ok(Rk45Result {
        t_final: t,
        y_final: y,
        t_history,
        y_history,
        n_steps,
        n_rejected,
        h_final: h,
    })
}

/// Solve to a specific time, returning only the final value.
///
/// # Errors
///
/// Returns [`Err`] if [`rk45_solve`] fails (see [`rk45_solve`] for conditions).
pub fn rk45_at<F>(
    f: &F,
    t_start: f64,
    t_end: f64,
    y0: &[f64],
    config: &Rk45Config,
) -> Result<Vec<f64>>
where
    F: Fn(f64, &[f64]) -> Vec<f64>,
{
    let result = rk45_solve(f, t_start, t_end, y0, config)?;
    Ok(result.y_final)
}

#[cfg(test)]
#[path = "rk45_tests.rs"]
mod tests;
