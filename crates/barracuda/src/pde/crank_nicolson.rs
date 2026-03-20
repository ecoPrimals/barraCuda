// SPDX-License-Identifier: AGPL-3.0-or-later
//! Crank-Nicolson scheme for 1D diffusion equations
//!
//! Implements the implicit Crank-Nicolson method for solving the heat equation
//! and similar parabolic PDEs. The scheme is unconditionally stable and
//! second-order accurate in both time and space.
//!
//! # The Heat Equation
//!
//! ```text
//! ∂u/∂t = α · ∂²u/∂x²
//! ```
//!
//! where α is the thermal diffusivity.
//!
//! # Crank-Nicolson Discretization
//!
//! ```text
//! (uⁿ⁺¹ᵢ - uⁿᵢ)/Δt = (α/2)[(uⁿ⁺¹ᵢ₊₁ - 2uⁿ⁺¹ᵢ + uⁿ⁺¹ᵢ₋₁)/Δx²
//!                       + (uⁿᵢ₊₁ - 2uⁿᵢ + uⁿᵢ₋₁)/Δx²]
//! ```
//!
//! This results in a tridiagonal system at each time step.
//!
//! # Applications
//!
//! - **TTM (Two-Temperature Model)**: Electron/lattice heat coupling
//! - **Thermal diffusion**: Heat transport in materials
//! - **Time-dependent Schrödinger**: Quantum dynamics (with i)
//!
//! # References
//!
//! - Crank, J. & Nicolson, P. (1947)
//! - Numerical Recipes, §19.2

/// GPU shader for Crank-Nicolson PDE solver (f64).
///
/// Entry points: `compute_rhs`, `build_matrix`, `apply_source`,
/// `adi_rhs_x_sweep`, `adi_rhs_y_sweep`, `compute_laplacian`.
///
/// Pairs with `cyclic_reduction_f64.wgsl` for O(log n) parallel tridiagonal solve.
pub const WGSL_CRANK_NICOLSON_F64: &str = include_str!("../shaders/pde/crank_nicolson_f64.wgsl");

use crate::error::{BarracudaError, Result};
use crate::ops::linalg::tridiagonal::tridiagonal_solve;

/// Configuration for the Crank-Nicolson solver.
#[derive(Debug, Clone)]
pub struct CrankNicolsonConfig {
    /// Thermal diffusivity α (units: length²/time)
    pub alpha: f64,
    /// Spatial step size Δx
    pub dx: f64,
    /// Time step size Δt
    pub dt: f64,
    /// Number of spatial grid points (including boundaries)
    pub nx: usize,
    /// Left boundary value (Dirichlet)
    pub left_bc: f64,
    /// Right boundary value (Dirichlet)
    pub right_bc: f64,
}

impl CrankNicolsonConfig {
    /// Create a new configuration.
    /// # Arguments
    /// * `alpha` - Thermal diffusivity
    /// * `dx` - Spatial step
    /// * `dt` - Time step
    /// * `nx` - Number of grid points
    /// # Example
    /// ```
    /// use barracuda::pde::CrankNicolsonConfig;
    /// let config = CrankNicolsonConfig::new(1.0, 0.01, 0.001, 101);
    /// ```
    #[must_use]
    pub fn new(alpha: f64, dx: f64, dt: f64, nx: usize) -> Self {
        Self {
            alpha,
            dx,
            dt,
            nx,
            left_bc: 0.0,
            right_bc: 0.0,
        }
    }

    /// Set boundary conditions.
    #[must_use]
    pub fn with_boundary_conditions(mut self, left: f64, right: f64) -> Self {
        self.left_bc = left;
        self.right_bc = right;
        self
    }

    /// Compute the Courant number r = α·Δt/Δx².
    /// The Crank-Nicolson scheme is unconditionally stable for all r,
    /// but r ~ 0.5 gives best accuracy.
    #[must_use]
    pub fn courant_number(&self) -> f64 {
        self.alpha * self.dt / (self.dx * self.dx)
    }

    /// Validate the configuration.
    /// # Errors
    /// Returns [`Err`] if `alpha` ≤ 0, `dx` ≤ 0, `dt` ≤ 0, or `nx` < 3.
    pub fn validate(&self) -> Result<()> {
        if self.alpha <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "alpha must be positive".to_string(),
            });
        }
        if self.dx <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "dx must be positive".to_string(),
            });
        }
        if self.dt <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "dt must be positive".to_string(),
            });
        }
        if self.nx < 3 {
            return Err(BarracudaError::InvalidInput {
                message: "nx must be >= 3".to_string(),
            });
        }
        Ok(())
    }
}

/// 1D Heat equation solver using Crank-Nicolson.
///
/// Solves ∂u/∂t = α·∂²u/∂x² with Dirichlet boundary conditions.
pub struct HeatEquation1D {
    config: CrankNicolsonConfig,
    /// Current solution (interior points only)
    u: Vec<f64>,
    /// Tridiagonal matrix diagonals (precomputed)
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    /// Courant number
    r: f64,
}

impl HeatEquation1D {
    /// Create a new solver with initial condition.
    /// # Arguments
    /// * `config` - Solver configuration
    /// * `initial` - Initial temperature profile (length nx)
    /// # Example
    /// ```
    /// use barracuda::pde::{CrankNicolsonConfig, HeatEquation1D};
    /// let config = CrankNicolsonConfig::new(1.0, 0.1, 0.01, 11)
    ///     .with_boundary_conditions(0.0, 0.0);
    /// // Initial condition: u(x,0) = sin(πx)
    /// let initial: Vec<f64> = (0..11)
    ///     .map(|i| (std::f64::consts::PI * i as f64 / 10.0).sin())
    ///     .collect();
    /// let solver = HeatEquation1D::new(config, &initial).unwrap();
    /// ```
    /// # Errors
    /// Returns [`Err`] if the configuration is invalid (see [`validate`](CrankNicolsonConfig::validate))
    /// or if `initial.len() != config.nx`.
    pub fn new(config: CrankNicolsonConfig, initial: &[f64]) -> Result<Self> {
        config.validate()?;

        if initial.len() != config.nx {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Initial condition length {} != nx {}",
                    initial.len(),
                    config.nx
                ),
            });
        }

        let r = config.courant_number();
        let n_interior = config.nx - 2; // Exclude boundary points

        // Build tridiagonal matrix for implicit part
        // (1 + r)uⁿ⁺¹ᵢ - (r/2)uⁿ⁺¹ᵢ₋₁ - (r/2)uⁿ⁺¹ᵢ₊₁ = RHS
        let a = vec![-r / 2.0; n_interior - 1]; // sub-diagonal
        let b = vec![1.0 + r; n_interior]; // main diagonal
        let c = vec![-r / 2.0; n_interior - 1]; // super-diagonal

        // Store interior points only
        let u = initial[1..config.nx - 1].to_vec();

        Ok(Self {
            config,
            u,
            a,
            b,
            c,
            r,
        })
    }

    /// Advance the solution by one time step.
    /// # Returns
    /// The new solution including boundary points.
    /// # Errors
    /// Returns [`Err`] if the tridiagonal solve fails (e.g., singular system or invalid dimensions).
    pub fn step(&mut self) -> Result<Vec<f64>> {
        let n = self.u.len();

        // Build RHS from explicit part
        // RHS_i = (1-r)uⁿᵢ + (r/2)uⁿᵢ₋₁ + (r/2)uⁿᵢ₊₁ + boundary terms
        let mut rhs = vec![0.0; n];

        for i in 0..n {
            let u_left = if i == 0 {
                self.config.left_bc
            } else {
                self.u[i - 1]
            };
            let u_right = if i == n - 1 {
                self.config.right_bc
            } else {
                self.u[i + 1]
            };

            rhs[i] = (1.0 - self.r) * self.u[i] + (self.r / 2.0) * (u_left + u_right);
        }

        // Add boundary contributions to first and last equations
        rhs[0] += (self.r / 2.0) * self.config.left_bc;
        rhs[n - 1] += (self.r / 2.0) * self.config.right_bc;

        // Solve the tridiagonal system
        self.u = tridiagonal_solve(&self.a, &self.b, &self.c, &rhs)?;

        // Return full solution including boundaries
        Ok(self.solution())
    }

    /// Advance the solution by multiple time steps.
    /// # Arguments
    /// * `n_steps` - Number of time steps
    /// # Returns
    /// The final solution including boundary points.
    /// # Errors
    /// Returns [`Err`] if any time step fails (see [`step`](Self::step)).
    pub fn advance(&mut self, n_steps: usize) -> Result<Vec<f64>> {
        for _ in 0..n_steps {
            self.step()?;
        }
        Ok(self.solution())
    }

    /// Get the current solution including boundary points.
    #[must_use]
    pub fn solution(&self) -> Vec<f64> {
        let mut full = Vec::with_capacity(self.config.nx);
        full.push(self.config.left_bc);
        full.extend_from_slice(&self.u);
        full.push(self.config.right_bc);
        full
    }

    /// Get the spatial grid points.
    #[must_use]
    pub fn grid(&self) -> Vec<f64> {
        (0..self.config.nx)
            .map(|i| i as f64 * self.config.dx)
            .collect()
    }

    /// Get the current time.
    #[must_use]
    pub fn config(&self) -> &CrankNicolsonConfig {
        &self.config
    }
}

/// Crank-Nicolson solver for general 1D diffusion with source terms.
///
/// Solves: ∂u/∂t = α·∂²u/∂x² + f(x,t)
pub struct CrankNicolson1D {
    heat: HeatEquation1D,
    time: f64,
}

impl CrankNicolson1D {
    /// Create a new solver.
    /// # Errors
    /// Returns [`Err`] if the configuration is invalid or `initial.len() != config.nx`.
    pub fn new(config: CrankNicolsonConfig, initial: &[f64]) -> Result<Self> {
        let heat = HeatEquation1D::new(config, initial)?;
        Ok(Self { heat, time: 0.0 })
    }

    /// Advance with optional source term.
    /// # Arguments
    /// * `source` - Source term f(x) at current time (interior points only)
    /// # Errors
    /// Returns [`Err`] if the tridiagonal solve fails (see
    /// [`HeatEquation1D::step`](HeatEquation1D::step)).
    pub fn step_with_source(&mut self, source: Option<&[f64]>) -> Result<Vec<f64>> {
        // Source term integration uses operator splitting — the source
        // contribution is applied after the diffusion step. For coupled
        // reaction-diffusion, a Strang splitting variant is recommended.
        let result = self.heat.step()?;
        self.time += self.heat.config.dt;

        // Add source contribution (forward Euler for simplicity)
        if let Some(src) = source {
            if src.len() == self.heat.u.len() {
                for (u, s) in self.heat.u.iter_mut().zip(src.iter()) {
                    *u += self.heat.config.dt * s;
                }
            }
        }

        Ok(result)
    }

    /// Get current time.
    #[must_use]
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get current solution.
    #[must_use]
    pub fn solution(&self) -> Vec<f64> {
        self.heat.solution()
    }
}

/// Perform a single Crank-Nicolson step (functional interface).
///
/// # Arguments
///
/// * `u` - Current solution (interior points only, length n-2)
/// * `alpha` - Diffusivity
/// * `dx` - Spatial step
/// * `dt` - Time step
/// * `left_bc` - Left boundary value
/// * `right_bc` - Right boundary value
///
/// # Returns
///
/// New solution at interior points.
///
/// # Errors
///
/// Returns [`Err`] if `u` is empty or the tridiagonal solve fails.
pub fn crank_nicolson_step(
    u: &[f64],
    alpha: f64,
    dx: f64,
    dt: f64,
    left_bc: f64,
    right_bc: f64,
) -> Result<Vec<f64>> {
    let n = u.len();
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "u must have at least 1 interior point".to_string(),
        });
    }

    let r = alpha * dt / (dx * dx);

    // Build tridiagonal system
    let a = vec![-r / 2.0; n - 1];
    let b = vec![1.0 + r; n];
    let c = vec![-r / 2.0; n - 1];

    // Build RHS
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        let u_left = if i == 0 { left_bc } else { u[i - 1] };
        let u_right = if i == n - 1 { right_bc } else { u[i + 1] };
        rhs[i] = (1.0 - r) * u[i] + (r / 2.0) * (u_left + u_right);
    }
    rhs[0] += (r / 2.0) * left_bc;
    rhs[n - 1] += (r / 2.0) * right_bc;

    tridiagonal_solve(&a, &b, &c, &rhs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_config_validation() {
        let config = CrankNicolsonConfig::new(1.0, 0.1, 0.01, 11);
        assert!(config.validate().is_ok());

        let bad = CrankNicolsonConfig::new(-1.0, 0.1, 0.01, 11);
        assert!(bad.validate().is_err());

        let bad2 = CrankNicolsonConfig::new(1.0, 0.1, 0.01, 2);
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_courant_number() {
        let config = CrankNicolsonConfig::new(1.0, 0.1, 0.005, 11);
        let r = config.courant_number();
        // r = 1.0 * 0.005 / 0.01 = 0.5
        assert!((r - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_heat_equation_steady_state() {
        // Steady state with constant boundaries: linear profile
        let config =
            CrankNicolsonConfig::new(1.0, 0.1, 0.01, 11).with_boundary_conditions(0.0, 1.0);

        // Start with initial condition
        let initial = vec![0.0; 11];
        let mut solver = HeatEquation1D::new(config, &initial).unwrap();

        // Advance many steps
        let _ = solver.advance(1000).unwrap();

        // Should approach linear profile
        let sol = solver.solution();
        let grid = solver.grid();
        for (i, (x, u)) in grid.iter().zip(sol.iter()).enumerate() {
            let expected = x / 1.0; // Linear from 0 to 1
            // Allow some error from finite time
            if i > 0 && i < 10 {
                assert!(
                    (u - expected).abs() < 0.1,
                    "At x={x}: u={u}, expected={expected}"
                );
            }
        }
    }

    #[test]
    fn test_heat_equation_decay() {
        // Test decay of initial sinusoidal perturbation
        // u(x,0) = sin(πx), α=1, L=1 → u(x,t) = exp(-π²t)sin(πx)
        let nx = 101;
        let dx = 1.0 / (nx - 1) as f64;
        let dt = 0.0001;
        let alpha = 1.0;

        let config = CrankNicolsonConfig::new(alpha, dx, dt, nx).with_boundary_conditions(0.0, 0.0);

        let initial: Vec<f64> = (0..nx)
            .map(|i| (PI * i as f64 / (nx - 1) as f64).sin())
            .collect();

        let mut solver = HeatEquation1D::new(config, &initial).unwrap();

        // Advance to t = 0.1
        let n_steps = 1000;
        let t_final = n_steps as f64 * dt;
        let _ = solver.advance(n_steps).unwrap();

        // Check decay at midpoint
        let sol = solver.solution();
        let mid_idx = nx / 2;
        let expected = (-PI * PI * t_final).exp() * 1.0; // sin(π/2) = 1

        assert!(
            (sol[mid_idx] - expected).abs() < 0.05,
            "At midpoint: got {}, expected {}",
            sol[mid_idx],
            expected
        );
    }

    #[test]
    fn test_crank_nicolson_step() {
        // Simple test of functional interface
        let u = vec![0.0, 1.0, 0.0]; // Interior points
        let result = crank_nicolson_step(&u, 1.0, 0.1, 0.005, 0.0, 0.0).unwrap();
        assert_eq!(result.len(), 3);

        // After one step, the peak should decrease
        assert!(result[1] < 1.0);
    }

    #[test]
    fn test_solver_with_source() {
        let config =
            CrankNicolsonConfig::new(1.0, 0.1, 0.01, 11).with_boundary_conditions(0.0, 0.0);

        let initial = vec![0.0; 11];
        let mut solver = CrankNicolson1D::new(config, &initial).unwrap();

        // Apply constant source
        let source = vec![1.0; 9]; // Interior points only
        let _ = solver.step_with_source(Some(&source)).unwrap();

        assert!(solver.time() > 0.0);
    }

    #[test]
    fn test_conservation_with_no_flux() {
        // With zero Dirichlet BCs and no initial gradient,
        // total "heat" should decrease
        let nx = 11;
        let config =
            CrankNicolsonConfig::new(1.0, 0.1, 0.01, nx).with_boundary_conditions(0.0, 0.0);

        let initial: Vec<f64> = (0..nx)
            .map(|i| if i == nx / 2 { 1.0 } else { 0.0 })
            .collect();

        let initial_sum: f64 = initial.iter().sum();

        let mut solver = HeatEquation1D::new(config, &initial).unwrap();
        let _ = solver.advance(10).unwrap();
        let final_sol = solver.solution();
        let final_sum: f64 = final_sol.iter().sum();

        // Heat should diffuse out through boundaries
        assert!(final_sum < initial_sum);
    }
}
