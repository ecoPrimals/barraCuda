// SPDX-License-Identifier: AGPL-3.0-or-later

//! Screened Coulomb (Yukawa) potential eigenvalue solver.
//!
//! Solves the radial Schrödinger equation for the screened Coulomb potential
//! V(r) = -Z·exp(-μr)/r using Sturm bisection on a discretized tridiagonal
//! Hamiltonian.
//!
//! # Physics
//!
//! For ℓ=0 (s-states), the radial equation with u(r) = r·R(r) is:
//!   -½ u'' + V(r) u = E u
//!
//! - μ=0: Coulomb potential (hydrogen-like)
//! - μ>0: Yukawa/screened Coulomb (e.g. Debye screening in plasmas)
//!
//! # References
//!
//! - Radial Schrödinger equation discretization (finite differences)
//! - Sturm sequence / bisection (spectral/tridiag)

use crate::error::{BarracudaError, Result};
use crate::spectral::find_all_eigenvalues;

/// Compute eigenvalues of the screened Coulomb (Yukawa) potential via Sturm bisection.
///
/// Discretizes the radial Schrödinger equation on a uniform grid, builds the
/// tridiagonal Hamiltonian, and uses Sturm bisection to find the lowest
/// `n_eigenvalues` eigenvalues.
///
/// # Arguments
///
/// * `z` - Nuclear charge (Z > 0). For hydrogen, Z=1.
/// * `mu` - Screening parameter (μ ≥ 0). μ=0 gives pure Coulomb.
/// * `n_grid` - Number of radial grid points (≥ 2).
/// * `r_max` - Maximum radial coordinate (must be > 0).
/// * `n_eigenvalues` - Number of lowest eigenvalues to return.
///
/// # Returns
///
/// Eigenvalues in ascending order. For hydrogen (Z=1, μ=0), these approximate
/// -0.5/n² (n = 1, 2, 3, ...) in atomic units.
///
/// # Errors
///
/// Returns `InvalidInput` if parameters are invalid.
pub fn screened_coulomb_eigenvalues(
    z: f64,
    mu: f64,
    n_grid: usize,
    r_max: f64,
    n_eigenvalues: usize,
) -> Result<Vec<f64>> {
    if z <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("screened_coulomb_eigenvalues requires z > 0, got {z}"),
        });
    }
    if mu < 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("screened_coulomb_eigenvalues requires mu >= 0, got {mu}"),
        });
    }
    if n_grid < 2 {
        return Err(BarracudaError::InvalidInput {
            message: format!("screened_coulomb_eigenvalues requires n_grid >= 2, got {n_grid}"),
        });
    }
    if r_max <= 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("screened_coulomb_eigenvalues requires r_max > 0, got {r_max}"),
        });
    }
    if n_eigenvalues == 0 {
        return Ok(Vec::new());
    }
    let n_eig = n_eigenvalues.min(n_grid);

    // Radial grid: r_i = (i+1)*h for i = 0..n_grid-1, with u(0)=0 boundary.
    // Starting at h avoids the 1/r singularity at the origin.
    let h = r_max / (n_grid + 1) as f64;

    // Tridiagonal Hamiltonian for -½ u'' + V(r) u = E u
    // Finite difference: -½ u'' ≈ (u_{i+1} - 2u_i + u_{i-1}) / (2h²)
    // Diagonal:     1/h² + V(r_i)
    // Off-diagonal: -1/(2h²)
    let kin_diag = 1.0 / (h * h);
    let kin_off = -0.5 / (h * h);

    let mut diagonal = Vec::with_capacity(n_grid);
    let mut off_diag = Vec::with_capacity(n_grid.saturating_sub(1));

    for i in 0..n_grid {
        let r = (i + 1) as f64 * h;
        let v = -z * (-mu * r).exp() / r;
        diagonal.push(kin_diag + v);
    }
    for _ in 0..n_grid.saturating_sub(1) {
        off_diag.push(kin_off);
    }

    let all_eigenvalues = find_all_eigenvalues(&diagonal, &off_diag);
    Ok(all_eigenvalues.into_iter().take(n_eig).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hydrogen_ground_state() {
        // Hydrogen (Z=1, μ=0): E_1 = -0.5
        let evals = screened_coulomb_eigenvalues(1.0, 0.0, 500, 30.0, 1).expect("valid params");
        assert_eq!(evals.len(), 1);
        let e1 = evals[0];
        assert!((e1 - (-0.5)).abs() < 0.01, "hydrogen E_1 ≈ -0.5, got {e1}");
    }

    #[test]
    fn hydrogen_first_levels() {
        // Hydrogen: E_n = -0.5/n² for n=1,2,3
        let evals = screened_coulomb_eigenvalues(1.0, 0.0, 800, 50.0, 5).expect("valid params");
        assert!(evals.len() >= 3);
        let expected = [-0.5, -0.125, -0.0556]; // -0.5/1², -0.5/2², -0.5/3²
        for (i, &exp) in expected.iter().enumerate() {
            let got = evals[i];
            assert!(
                (got - exp).abs() < 0.02,
                "n={}, expected ≈ {exp}, got {got}",
                i + 1
            );
        }
    }

    #[test]
    fn screened_less_bound_than_unscreened() {
        // Screened potential (μ>0) gives less negative eigenvalues than Coulomb
        let coulomb = screened_coulomb_eigenvalues(1.0, 0.0, 500, 30.0, 3).expect("coulomb");
        let screened = screened_coulomb_eigenvalues(1.0, 0.5, 500, 30.0, 3).expect("screened");
        for (ec, es) in coulomb.iter().zip(screened.iter()) {
            assert!(es > ec, "screened eigenvalue {es} should be > coulomb {ec}");
        }
    }

    #[test]
    fn invalid_z_rejected() {
        let r = screened_coulomb_eigenvalues(0.0, 0.0, 100, 20.0, 1);
        assert!(r.is_err());
    }

    #[test]
    fn invalid_mu_rejected() {
        let r = screened_coulomb_eigenvalues(1.0, -0.1, 100, 20.0, 1);
        assert!(r.is_err());
    }

    #[test]
    fn empty_eigenvalues_ok() {
        let evals = screened_coulomb_eigenvalues(1.0, 0.0, 100, 20.0, 0).expect("ok");
        assert!(evals.is_empty());
    }
}
