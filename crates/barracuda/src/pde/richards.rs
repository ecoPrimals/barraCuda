// SPDX-License-Identifier: AGPL-3.0-or-later

//! 1D Richards Equation Solver — Unsaturated zone water flow.
//!
//! Solves the mixed-form Richards equation:
//!   ∂θ/∂t = ∂/∂z [K(h) (∂h/∂z + 1)]
//!
//! where:
//!   h = pressure head (negative in unsaturated zone)
//!   θ = volumetric water content = θ(h) via van Genuchten
//!   K = hydraulic conductivity = K(h) via van Genuchten-Mualem
//!   z = depth (positive downward)
//!
//! Uses Picard iteration with Crank-Nicolson time discretization and
//! Thomas algorithm for the resulting tridiagonal system.
//!
//! Provenance: airSpring precision agriculture → toadStool absorption

use crate::error::{BarracudaError, Result};

/// WGSL kernel for Van Genuchten soil hydraulic model (f64).
pub const WGSL_VAN_GENUCHTEN_F64: &str = include_str!("../shaders/science/van_genuchten_f64.wgsl");

/// Guard against division by zero in inter-node harmonic means.
const HARMONIC_MEAN_GUARD: f64 = 1e-30;

/// Minimum moisture capacity to prevent singular tridiagonal systems.
const MIN_CAPACITY: f64 = 1e-10;

/// Van Genuchten soil hydraulic parameters.
#[derive(Debug, Clone, Copy)]
pub struct SoilParams {
    /// Saturated water content (dimensionless, typically 0.3–0.5)
    pub theta_s: f64,
    /// Residual water content (dimensionless, typically 0.03–0.1)
    pub theta_r: f64,
    /// Inverse of air-entry pressure (1/cm, typically 0.01–0.5)
    pub alpha: f64,
    /// Pore-size distribution parameter (dimensionless, typically 1.1–3.0)
    pub n: f64,
    /// Saturated hydraulic conductivity (cm/s)
    pub k_sat: f64,
}

impl SoilParams {
    /// Sandy loam (Carsel & Parrish 1988, Table 3).
    pub const SANDY_LOAM: Self = Self {
        theta_s: 0.41,
        theta_r: 0.065,
        alpha: 0.075,
        n: 1.89,
        k_sat: 1.228e-3,
    };
    /// Silt loam (Carsel & Parrish 1988, Table 3).
    pub const SILT_LOAM: Self = Self {
        theta_s: 0.45,
        theta_r: 0.067,
        alpha: 0.020,
        n: 1.41,
        k_sat: 1.25e-4,
    };
    /// Clay loam (Carsel & Parrish 1988, Table 3).
    pub const CLAY_LOAM: Self = Self {
        theta_s: 0.41,
        theta_r: 0.095,
        alpha: 0.019,
        n: 1.31,
        k_sat: 7.22e-5,
    };
    /// Sand (Carsel & Parrish 1988, Table 3).
    pub const SAND: Self = Self {
        theta_s: 0.43,
        theta_r: 0.045,
        alpha: 0.145,
        n: 2.68,
        k_sat: 8.25e-3,
    };
    /// Clay (Carsel & Parrish 1988, Table 3).
    pub const CLAY: Self = Self {
        theta_s: 0.38,
        theta_r: 0.068,
        alpha: 0.008,
        n: 1.09,
        k_sat: 5.56e-5,
    };
    /// Loam (Carsel & Parrish 1988, Table 3).
    pub const LOAM: Self = Self {
        theta_s: 0.43,
        theta_r: 0.078,
        alpha: 0.036,
        n: 1.56,
        k_sat: 2.89e-4,
    };
    /// Silty clay loam (Carsel & Parrish 1988, Table 3).
    pub const SILTY_CLAY_LOAM: Self = Self {
        theta_s: 0.43,
        theta_r: 0.089,
        alpha: 0.010,
        n: 1.23,
        k_sat: 1.94e-5,
    };
    /// Loamy sand (Carsel & Parrish 1988, Table 3).
    pub const LOAMY_SAND: Self = Self {
        theta_s: 0.41,
        theta_r: 0.057,
        alpha: 0.124,
        n: 2.28,
        k_sat: 4.05e-3,
    };

    /// Van Genuchten m parameter: m = 1 - 1/n
    fn m(&self) -> f64 {
        1.0 - 1.0 / self.n
    }

    /// Effective saturation `S_e(h)` via van Genuchten.
    #[must_use]
    pub fn effective_saturation(&self, h: f64) -> f64 {
        if h >= 0.0 {
            return 1.0;
        }
        let ah = (self.alpha * (-h)).max(0.0);
        (1.0 + ah.powf(self.n)).powf(-self.m())
    }

    /// Water content θ(h) via van Genuchten.
    #[must_use]
    pub fn theta(&self, h: f64) -> f64 {
        (self.theta_s - self.theta_r).mul_add(self.effective_saturation(h), self.theta_r)
    }

    /// Specific moisture capacity C(h) = dθ/dh via van Genuchten.
    #[must_use]
    pub fn capacity(&self, h: f64) -> f64 {
        if h >= 0.0 {
            return 0.0;
        }
        let m = self.m();
        let ah = self.alpha * (-h);
        let ah_n = ah.powf(self.n);
        let denom = 1.0 + ah_n;

        (self.theta_s - self.theta_r) * self.alpha * self.n * m * ah.powf(self.n - 1.0)
            / denom.powf(m + 1.0)
    }

    /// Hydraulic conductivity K(h) via van Genuchten-Mualem model.
    #[must_use]
    pub fn conductivity(&self, h: f64) -> f64 {
        let se = self.effective_saturation(h);
        let m = self.m();
        self.k_sat * se.sqrt() * (1.0 - (1.0 - se.powf(1.0 / m)).powf(m)).powi(2)
    }
}

/// Boundary condition for Richards equation.
#[derive(Debug, Clone, Copy)]
pub enum RichardsBc {
    /// Fixed pressure head (cm).
    PressureHead(f64),
    /// Fixed flux (cm/s, positive = downward infiltration).
    Flux(f64),
}

/// Configuration for the Richards solver.
#[derive(Debug, Clone)]
pub struct RichardsConfig {
    /// Soil hydraulic parameters.
    pub soil: SoilParams,
    /// Grid spacing (cm).
    pub dz: f64,
    /// Time step (s).
    pub dt: f64,
    /// Number of grid nodes (including boundaries).
    pub n_nodes: usize,
    /// Maximum Picard iterations per time step.
    pub max_picard_iter: usize,
    /// Picard convergence tolerance (cm pressure head).
    pub picard_tol: f64,
}

impl RichardsConfig {
    /// Validates configuration parameters (nodes, spacing, time step, soil params).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn validate(&self) -> Result<()> {
        if self.n_nodes < 3 {
            return Err(BarracudaError::InvalidInput {
                message: "Richards solver requires at least 3 nodes".into(),
            });
        }
        if self.dz <= 0.0 || self.dt <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "dz and dt must be positive".into(),
            });
        }
        if self.soil.n <= 1.0 {
            return Err(BarracudaError::InvalidInput {
                message: "van Genuchten n must be > 1".into(),
            });
        }
        Ok(())
    }
}

/// Result of a Richards simulation.
#[derive(Debug, Clone)]
pub struct RichardsResult {
    /// Pressure head profile at final time (cm).
    pub h: Vec<f64>,
    /// Water content profile at final time (dimensionless).
    pub theta: Vec<f64>,
    /// Total Picard iterations across all time steps.
    pub total_picard_iterations: usize,
    /// Time steps completed.
    pub time_steps_completed: usize,
}

/// 1D Richards equation solver (CPU, f64).
///
/// Uses Picard linearization with Crank-Nicolson:
/// - Evaluates K and C at the current iterate
/// - Assembles and solves the tridiagonal system
/// - Iterates until convergence or max iterations
///
/// # Errors
///
/// Returns [`Err`] if config validation fails, h0 length mismatch, or tridiagonal solve fails.
pub fn solve_richards(
    config: &RichardsConfig,
    h0: &[f64],
    n_steps: usize,
    top_bc: RichardsBc,
    bottom_bc: RichardsBc,
) -> Result<RichardsResult> {
    config.validate()?;
    let n = config.n_nodes;

    if h0.len() != n {
        return Err(BarracudaError::InvalidInput {
            message: format!("h0 length {} != n_nodes {}", h0.len(), n),
        });
    }

    let soil = &config.soil;
    let dz = config.dz;
    let dt = config.dt;

    let mut h = h0.to_vec();
    let mut total_picard = 0usize;
    let dz2 = dz * dz;

    // Preallocate Picard iteration buffers once (not per-step or per-iteration)
    let mut h_old = vec![0.0; n];
    let mut theta_old = vec![0.0; n];
    let mut k_buf = vec![0.0; n];
    let mut c_buf = vec![0.0; n];
    let mut k_half = vec![0.0; n - 1];
    let mut a_tri = vec![0.0; n];
    let mut b_tri = vec![0.0; n];
    let mut c_tri = vec![0.0; n];
    let mut d_vec = vec![0.0; n];

    for _step in 0..n_steps {
        h_old.copy_from_slice(&h);
        for (i, &hi) in h_old.iter().enumerate() {
            theta_old[i] = soil.theta(hi);
        }

        for _picard in 0..config.max_picard_iter {
            total_picard += 1;

            for (i, &hi) in h.iter().enumerate() {
                k_buf[i] = soil.conductivity(hi);
                c_buf[i] = soil.capacity(hi);
            }

            for i in 0..n - 1 {
                k_half[i] =
                    2.0 * k_buf[i] * k_buf[i + 1] / (k_buf[i] + k_buf[i + 1] + HARMONIC_MEAN_GUARD);
            }

            // Zero the tridiagonal vectors for this iteration
            a_tri.fill(0.0);
            b_tri.fill(0.0);
            c_tri.fill(0.0);
            d_vec.fill(0.0);

            for i in 1..n - 1 {
                let ci_max = c_buf[i].max(MIN_CAPACITY);
                let coeff_l = k_half[i - 1] / dz2;
                let coeff_r = k_half[i] / dz2;

                a_tri[i] = -0.5 * dt * coeff_l;
                c_tri[i] = -0.5 * dt * coeff_r;
                b_tri[i] = (0.5 * dt).mul_add(coeff_l + coeff_r, ci_max);

                d_vec[i] = (0.5 * dt * coeff_r).mul_add(
                    h_old[i + 1],
                    (0.5 * dt * (coeff_l + coeff_r)).mul_add(
                        -h_old[i],
                        ci_max * h_old[i] + 0.5 * dt * coeff_l * h_old[i - 1],
                    ),
                ) + dt * (k_half[i] - k_half[i - 1]) / dz;

                d_vec[i] += theta_old[i] - soil.theta(h[i]);
            }

            match top_bc {
                RichardsBc::PressureHead(h_top) => {
                    b_tri[0] = 1.0;
                    d_vec[0] = h_top;
                }
                RichardsBc::Flux(q_top) => {
                    let coeff_r = k_half[0] / dz2;
                    let ci_max = c_buf[0].max(MIN_CAPACITY);
                    b_tri[0] = (0.5 * dt).mul_add(coeff_r, ci_max);
                    c_tri[0] = -0.5 * dt * coeff_r;
                    d_vec[0] = (0.5 * dt * coeff_r)
                        .mul_add(h_old[1], ci_max * h_old[0] - 0.5 * dt * coeff_r * h_old[0])
                        + dt * q_top / dz;
                }
            }

            match bottom_bc {
                RichardsBc::PressureHead(h_bot) => {
                    b_tri[n - 1] = 1.0;
                    d_vec[n - 1] = h_bot;
                }
                RichardsBc::Flux(q_bot) => {
                    let coeff_l = k_half[n - 2] / dz2;
                    let ci_max = c_buf[n - 1].max(MIN_CAPACITY);
                    a_tri[n - 1] = -0.5 * dt * coeff_l;
                    b_tri[n - 1] = (0.5 * dt).mul_add(coeff_l, ci_max);
                    d_vec[n - 1] = (0.5 * dt * coeff_l).mul_add(
                        -h_old[n - 1],
                        ci_max * h_old[n - 1] + 0.5 * dt * coeff_l * h_old[n - 2],
                    ) + dt * q_bot / dz;
                }
            }

            let h_new = thomas_solve(&a_tri, &b_tri, &c_tri, &d_vec);

            let max_diff = h_new
                .iter()
                .zip(h.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f64, f64::max);

            h = h_new;

            if max_diff < config.picard_tol {
                break;
            }
        }
    }

    let theta: Vec<f64> = h.iter().map(|&hi| soil.theta(hi)).collect();

    Ok(RichardsResult {
        h,
        theta,
        total_picard_iterations: total_picard,
        time_steps_completed: n_steps,
    })
}

/// Thomas algorithm for tridiagonal system Ax = d.
fn thomas_solve(a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut c_star = vec![0.0; n];
    let mut d_star = vec![0.0; n];
    let mut x = vec![0.0; n];

    // Forward sweep
    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for i in 1..n {
        let m = a[i].mul_add(-c_star[i - 1], b[i]);
        c_star[i] = c[i] / m;
        d_star[i] = a[i].mul_add(-d_star[i - 1], d[i]) / m;
    }

    // Back substitution
    x[n - 1] = d_star[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = c_star[i].mul_add(-x[i + 1], d_star[i]);
    }

    x
}

#[cfg(test)]
mod tests {
    use super::*;

    const _: () = {
        assert!(SoilParams::SAND.n > SoilParams::CLAY.n);
        assert!(SoilParams::SAND.k_sat > SoilParams::CLAY.k_sat);
        assert!(SoilParams::SANDY_LOAM.alpha > SoilParams::SILT_LOAM.alpha);
        assert!(SoilParams::LOAMY_SAND.n > 2.0);
        assert!(SoilParams::SILTY_CLAY_LOAM.n < 1.5);
    };

    #[test]
    fn test_van_genuchten_saturation() {
        let soil = SoilParams::SANDY_LOAM;
        assert!((soil.effective_saturation(0.0) - 1.0).abs() < 1e-12);
        assert!((soil.effective_saturation(10.0) - 1.0).abs() < 1e-12);
        let se = soil.effective_saturation(-100.0);
        assert!(se > 0.0 && se < 1.0, "S_e(-100cm) = {se}");
    }

    #[test]
    fn test_van_genuchten_conductivity() {
        let soil = SoilParams::SANDY_LOAM;
        assert!((soil.conductivity(0.0) - soil.k_sat).abs() < 1e-12);
        let k_dry = soil.conductivity(-1000.0);
        assert!(k_dry < soil.k_sat * 0.01, "K(-1000cm) should be << K_sat");
    }

    #[test]
    fn test_steady_state_uniform() {
        let soil = SoilParams::SANDY_LOAM;
        let n = 20;
        let h0 = vec![-50.0; n]; // uniform initial condition

        let config = RichardsConfig {
            soil,
            dz: 5.0,
            dt: 60.0,
            n_nodes: n,
            max_picard_iter: 20,
            picard_tol: 1e-6,
        };

        let result = solve_richards(
            &config,
            &h0,
            100,
            RichardsBc::PressureHead(-50.0),
            RichardsBc::PressureHead(-50.0),
        )
        .unwrap();

        // Uniform BCs + uniform initial → should stay near -50
        for &hi in &result.h {
            assert!((hi - (-50.0)).abs() < 1.0, "h = {hi}, expected near -50");
        }
    }

    #[test]
    fn test_infiltration_wets_top() {
        let soil = SoilParams::SANDY_LOAM;
        let n = 30;
        let h0 = vec![-200.0; n]; // dry initial condition

        let config = RichardsConfig {
            soil,
            dz: 2.0,
            dt: 10.0,
            n_nodes: n,
            max_picard_iter: 30,
            picard_tol: 1e-4,
        };

        let result = solve_richards(
            &config,
            &h0,
            200,
            RichardsBc::PressureHead(-10.0),  // wet top
            RichardsBc::PressureHead(-200.0), // dry bottom
        )
        .unwrap();

        // Top should be wetter (less negative) than bottom
        assert!(
            result.h[0] > result.h[n - 1],
            "top h ({}) should be > bottom h ({})",
            result.h[0],
            result.h[n - 1]
        );
        assert!(
            result.theta[0] > result.theta[n - 1],
            "top θ should be > bottom θ"
        );
    }
}
