// SPDX-License-Identifier: AGPL-3.0-or-later

//! RHMC Hybrid Monte Carlo — pseudofermion heatbath, action, and force.
//!
//! CPU reference implementation for RHMC trajectories with fractional
//! fermion determinant powers. Uses [`RationalApproximation`] from
//! [`super::rhmc`] and multi-shift CG for efficient shifted solves.
//!
//! Absorbed from hotSpring v0.64 `lattice/rhmc.rs` (Mar 2026).
//!
//! # References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — RHMC algorithm
//! - Gottlieb et al., PRD 35, 2531 (1987) — pseudofermion HMC

use super::cpu_complex::Complex64;
use super::cpu_dirac::FermionField;
use super::cpu_su3::Su3Matrix;
use super::rhmc::{RationalApproximation, multi_shift_cg_solve};
use super::wilson::Lattice;

/// RHMC configuration for a single fermion taste/flavor sector.
#[derive(Clone, Debug)]
pub struct RhmcFermionConfig {
    /// Mass for this sector.
    pub mass: f64,
    /// Power of the determinant (e.g. 1/4 for one rooted staggered taste).
    pub det_power: f64,
    /// Rational approximation for the action: x^{`det_power`}.
    pub action_approx: RationalApproximation,
    /// Rational approximation for the heatbath: x^{-`det_power`/2}.
    pub heatbath_approx: RationalApproximation,
    /// Rational approximation for the force: x^{`det_power`} (derivative form).
    pub force_approx: RationalApproximation,
}

/// RHMC HMC configuration.
#[derive(Clone, Debug)]
pub struct RhmcConfig {
    /// Fermion sectors (one per flavor group).
    pub sectors: Vec<RhmcFermionConfig>,
    /// Gauge coupling.
    pub beta: f64,
    /// MD step size.
    pub dt: f64,
    /// Number of MD steps.
    pub n_md_steps: usize,
    /// CG tolerance.
    pub cg_tol: f64,
    /// CG max iterations.
    pub cg_max_iter: usize,
}

impl RhmcConfig {
    /// Nf=2 configuration: one rooted staggered field with det(D†D)^{1/2}.
    #[must_use]
    pub fn nf2(mass: f64, beta: f64) -> Self {
        Self {
            sectors: vec![RhmcFermionConfig {
                mass,
                det_power: 0.5,
                action_approx: RationalApproximation::sqrt_8pole(),
                heatbath_approx: RationalApproximation::inv_sqrt_8pole(),
                force_approx: RationalApproximation::sqrt_8pole(),
            }],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        }
    }

    /// Nf=2+1 configuration: light (u,d) with det^{1/2} + strange with det^{1/4}.
    #[must_use]
    pub fn nf2p1(light_mass: f64, strange_mass: f64, beta: f64) -> Self {
        Self {
            sectors: vec![
                RhmcFermionConfig {
                    mass: light_mass,
                    det_power: 0.5,
                    action_approx: RationalApproximation::sqrt_8pole(),
                    heatbath_approx: RationalApproximation::inv_sqrt_8pole(),
                    force_approx: RationalApproximation::sqrt_8pole(),
                },
                RhmcFermionConfig {
                    mass: strange_mass,
                    det_power: 0.25,
                    action_approx: RationalApproximation::fourth_root_8pole(),
                    heatbath_approx: RationalApproximation::inv_fourth_root_8pole(),
                    force_approx: RationalApproximation::fourth_root_8pole(),
                },
            ],
            beta,
            dt: 0.01,
            n_md_steps: 50,
            cg_tol: 1e-8,
            cg_max_iter: 5000,
        }
    }
}

/// RHMC pseudofermion heatbath: generate φ = (D†D)^{-p/2} η where η ~ N(0,1).
///
/// Uses the rational approximation for x^{-p/2} and multi-shift CG.
pub fn rhmc_heatbath(
    lattice: &Lattice,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
    seed: &mut u64,
) -> (FermionField, usize) {
    let vol = lattice.volume();
    let approx = &config.heatbath_approx;

    let mut eta = FermionField::zeros(vol);
    for site in &mut eta.data {
        for c in site.iter_mut() {
            let re = super::constants::lcg_gaussian(seed);
            let im = super::constants::lcg_gaussian(seed);
            *c = Complex64::new(re, im);
        }
    }

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        &eta,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut phi = FermionField::zeros(vol);
    for i in 0..vol {
        for c in 0..3 {
            let mut val = eta.data[i][c].scale(approx.alpha_0);
            for (s, x_s) in solutions.iter().enumerate() {
                val += x_s.data[i][c].scale(approx.alpha[s]);
            }
            phi.data[i][c] = val;
        }
    }

    (phi, result.iterations)
}

/// RHMC fermion action: `S_f` = φ† r(p)(D†D) φ.
///
/// Uses the rational approximation for x^p and multi-shift CG.
#[must_use]
pub fn rhmc_fermion_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (f64, usize) {
    let approx = &config.action_approx;

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        phi,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut action = approx.alpha_0 * phi.dot(phi).re;
    for (s, x_s) in solutions.iter().enumerate() {
        action += approx.alpha[s] * phi.dot(x_s).re;
    }

    (action, result.iterations)
}

/// RHMC fermion force: `dS_f`/dU = Σ `α_i` · d/dU \[φ† (D†D + `σ_i`)⁻¹ φ\].
///
/// Each term in the sum is a standard pseudofermion force evaluated at the
/// shifted CG solution. Returns the total force summed over all poles.
#[must_use]
pub fn rhmc_fermion_force(
    lattice: &Lattice,
    phi: &FermionField,
    config: &RhmcFermionConfig,
    cg_tol: f64,
    cg_max_iter: usize,
) -> (Vec<Su3Matrix>, usize) {
    let vol = lattice.volume();
    let approx = &config.force_approx;

    let (solutions, result) = multi_shift_cg_solve(
        lattice,
        phi,
        config.mass,
        &approx.sigma,
        cg_tol,
        cg_max_iter,
    );

    let mut total_force = vec![Su3Matrix::ZERO; vol * 4];

    for (s, x_s) in solutions.iter().enumerate() {
        let f_s = super::pseudofermion::pseudofermion_force(lattice, x_s, config.mass);
        for (tf, fs) in total_force.iter_mut().zip(&f_s) {
            *tf = *tf + fs.scale(approx.alpha[s]);
        }
    }

    (total_force, result.iterations)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rhmc_config_nf2_has_one_sector() {
        let config = RhmcConfig::nf2(0.1, 6.0);
        assert_eq!(config.sectors.len(), 1);
        assert!((config.sectors[0].det_power - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn rhmc_config_nf2p1_has_two_sectors() {
        let config = RhmcConfig::nf2p1(0.01, 0.1, 6.0);
        assert_eq!(config.sectors.len(), 2);
        assert!((config.sectors[0].det_power - 0.5).abs() < f64::EPSILON);
        assert!((config.sectors[1].det_power - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn rhmc_heatbath_produces_nonzero_field() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let config = &RhmcConfig::nf2(0.5, 6.0).sectors[0];
        let mut seed = 42_u64;
        let (phi, iters) = rhmc_heatbath(&lat, config, 1e-6, 500, &mut seed);
        assert!(phi.norm_sq() > 0.0, "heatbath should produce non-zero φ");
        assert!(iters > 0, "should take at least one CG iteration");
    }

    #[test]
    fn rhmc_fermion_action_positive() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let config = &RhmcConfig::nf2(0.5, 6.0).sectors[0];
        let mut seed = 123_u64;
        let (phi, _) = rhmc_heatbath(&lat, config, 1e-6, 500, &mut seed);
        let (action, _) = rhmc_fermion_action(&lat, &phi, config, 1e-6, 500);
        assert!(
            action > 0.0,
            "RHMC fermion action should be positive: {action}"
        );
    }

    #[test]
    fn rhmc_fermion_force_has_correct_size() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let config = &RhmcConfig::nf2(0.5, 6.0).sectors[0];
        let mut seed = 77_u64;
        let (phi, _) = rhmc_heatbath(&lat, config, 1e-6, 500, &mut seed);
        let (force, _) = rhmc_fermion_force(&lat, &phi, config, 1e-6, 500);
        assert_eq!(force.len(), vol * 4, "force should have vol*4 entries");
    }
}
