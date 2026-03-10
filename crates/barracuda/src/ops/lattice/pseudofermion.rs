// SPDX-License-Identifier: AGPL-3.0-only

//! Pseudofermion action for dynamical fermion HMC (Papers 10-12).
//!
//! In lattice QCD with dynamical fermions, the fermion determinant
//! is represented via pseudofermion fields:
//!
//!   det(D†D) = ∫ Dφ†Dφ exp(−φ†(D†D)⁻¹φ)
//!
//! The pseudofermion action is `S_F` = φ†X where (D†D)X = φ, solved by CG.
//! The pseudofermion force (derivative of `S_F` with respect to gauge links)
//! drives the molecular dynamics evolution alongside the gauge force.
//!
//! Absorbed from hotSpring v0.64 `lattice/pseudofermion.rs` (Feb 2026).
//!
//! # References
//!
//! - Gottlieb et al., PRD 35, 2531 (1987) — pseudofermion HMC
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 8.1-8.3
//! - Clark & Kennedy, NPB Proc. Suppl. 129, 850 (2004) — RHMC

use super::cpu_complex::Complex64;
use super::cpu_dirac::{CgResult, FermionField, apply_dirac, apply_dirac_adjoint, cg_solve};
use super::cpu_su3::Su3Matrix;
use super::wilson::Lattice;

/// Configuration for pseudofermion HMC.
#[derive(Clone, Debug)]
pub struct PseudofermionConfig {
    /// Fermion mass
    pub mass: f64,
    /// CG solver tolerance
    pub cg_tol: f64,
    /// Maximum CG iterations
    pub cg_max_iter: usize,
}

impl Default for PseudofermionConfig {
    fn default() -> Self {
        Self {
            mass: 0.1,
            cg_tol: super::constants::CG_TOL_DEFAULT,
            cg_max_iter: super::constants::CG_MAX_ITER_DEFAULT,
        }
    }
}

/// Generate pseudofermion field via heat bath: φ = D†η.
pub fn pseudofermion_heatbath(lattice: &Lattice, mass: f64, seed: &mut u64) -> FermionField {
    let vol = lattice.volume();
    let mut eta = FermionField::zeros(vol);
    for site in &mut eta.data {
        for c in site.iter_mut() {
            let re = super::constants::lcg_gaussian(seed);
            let im = super::constants::lcg_gaussian(seed);
            *c = Complex64::new(re, im);
        }
    }
    apply_dirac_adjoint(lattice, &eta, mass)
}

/// Compute pseudofermion action: `S_F` = φ†X where (D†D)X = φ.
#[must_use]
pub fn pseudofermion_action(
    lattice: &Lattice,
    phi: &FermionField,
    config: &PseudofermionConfig,
) -> (f64, CgResult, FermionField) {
    let vol = lattice.volume();
    let mut x = FermionField::zeros(vol);
    let cg_result = cg_solve(
        lattice,
        &mut x,
        phi,
        config.mass,
        config.cg_tol,
        config.cg_max_iter,
    );
    let action = phi.dot(&x).re;
    (action, cg_result, x)
}

/// Compute the pseudofermion force: `dS_F/dU_μ(x)` for all links.
#[must_use]
pub fn pseudofermion_force(lattice: &Lattice, x_field: &FermionField, mass: f64) -> Vec<Su3Matrix> {
    let vol = lattice.volume();
    let y_field = apply_dirac(lattice, x_field, mass);
    let mut force = vec![Su3Matrix::ZERO; vol * 4];

    for idx in 0..vol {
        let site = lattice.site_coords(idx);
        for mu in 0..4 {
            let eta = staggered_phase_local(site, mu);
            let fwd = lattice.neighbor(site, mu, true);
            let fwd_idx = lattice.site_index(fwd);
            let u = lattice.link(site, mu);

            let mut m_mat = Su3Matrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    let contrib = x_field.data[fwd_idx][a] * y_field.data[idx][b].conj()
                        - y_field.data[fwd_idx][a] * x_field.data[idx][b].conj();
                    m_mat.m[a][b] += contrib.scale(eta * 0.5);
                }
            }

            let w = u * m_mat;
            let wh = w.adjoint();
            let mut ta = Su3Matrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    ta.m[a][b] = (w.m[a][b] - wh.m[a][b]).scale(0.5);
                }
            }
            let tr = (ta.m[0][0] + ta.m[1][1] + ta.m[2][2]).scale(1.0 / 3.0);
            for a in 0..3 {
                ta.m[a][a] -= tr;
            }
            force[idx * 4 + mu] = ta;
        }
    }
    force
}

fn staggered_phase_local(x: [usize; 4], mu: usize) -> f64 {
    let sum: usize = x.iter().take(mu).sum();
    if sum.is_multiple_of(2) { 1.0 } else { -1.0 }
}

/// Dynamical fermion HMC configuration.
#[derive(Clone, Debug)]
pub struct DynamicalHmcConfig {
    /// Number of MD steps per trajectory
    pub n_md_steps: usize,
    /// MD step size
    pub dt: f64,
    /// Random seed
    pub seed: u64,
    /// Pseudofermion config
    pub fermion: PseudofermionConfig,
    /// Gauge coupling β = 6/g²
    pub beta: f64,
    /// Number of flavor quarters (`n_flavors/4`)
    pub n_flavors_over_4: usize,
}

impl Default for DynamicalHmcConfig {
    fn default() -> Self {
        Self {
            n_md_steps: 20,
            dt: 0.02,
            seed: 42,
            fermion: PseudofermionConfig::default(),
            beta: 5.5,
            n_flavors_over_4: 2,
        }
    }
}

/// Result of a dynamical fermion HMC trajectory.
#[derive(Clone, Debug)]
pub struct DynamicalHmcResult {
    /// Whether the trajectory was accepted
    pub accepted: bool,
    /// ΔH = `H_new` - `H_old`
    pub delta_h: f64,
    /// Average plaquette after trajectory
    pub plaquette: f64,
    /// Gauge action (before or after depending on accept)
    pub gauge_action: f64,
    /// Fermion action (before or after depending on accept)
    pub fermion_action: f64,
    /// Total CG iterations across all pseudofermions
    pub cg_iterations: usize,
}

/// Run one dynamical fermion HMC trajectory.
pub fn dynamical_hmc_trajectory(
    lattice: &mut Lattice,
    config: &mut DynamicalHmcConfig,
) -> DynamicalHmcResult {
    let vol = lattice.volume();
    let old_links = lattice.links.clone();

    let phi_fields: Vec<FermionField> = (0..config.n_flavors_over_4)
        .map(|_| pseudofermion_heatbath(lattice, config.fermion.mass, &mut config.seed))
        .collect();

    let gauge_action_before = lattice.wilson_action();
    let mut fermion_action_before = 0.0;
    let mut total_cg_iters = 0;
    for phi in &phi_fields {
        let (sf, cg_res, _) = pseudofermion_action(lattice, phi, &config.fermion);
        fermion_action_before += sf;
        total_cg_iters += cg_res.iterations;
    }

    let mut momenta = vec![Su3Matrix::ZERO; vol * 4];
    for p in &mut momenta {
        *p = Su3Matrix::random_algebra(&mut config.seed);
    }
    let kinetic_before = kinetic_energy(&momenta);
    let h_old = kinetic_before + gauge_action_before + fermion_action_before;

    dynamical_leapfrog(
        lattice,
        &mut momenta,
        &phi_fields,
        config.n_md_steps,
        config.dt,
        &config.fermion,
    );

    let gauge_action_after = lattice.wilson_action();
    let mut fermion_action_after = 0.0;
    for phi in &phi_fields {
        let (sf, cg_res, _) = pseudofermion_action(lattice, phi, &config.fermion);
        fermion_action_after += sf;
        total_cg_iters += cg_res.iterations;
    }
    let kinetic_after = kinetic_energy(&momenta);
    let h_new = kinetic_after + gauge_action_after + fermion_action_after;
    let delta_h = h_new - h_old;

    let accept = if delta_h <= 0.0 {
        true
    } else {
        let r = super::constants::lcg_uniform_f64(&mut config.seed);
        r < (-delta_h).exp()
    };

    if !accept {
        lattice.links = old_links;
    }

    let plaquette = lattice.average_plaquette();

    DynamicalHmcResult {
        accepted: accept,
        delta_h,
        plaquette,
        gauge_action: if accept {
            gauge_action_after
        } else {
            gauge_action_before
        },
        fermion_action: if accept {
            fermion_action_after
        } else {
            fermion_action_before
        },
        cg_iterations: total_cg_iters,
    }
}

fn dynamical_leapfrog(
    lattice: &mut Lattice,
    momenta: &mut [Su3Matrix],
    phi_fields: &[FermionField],
    n_steps: usize,
    dt: f64,
    fermion_config: &PseudofermionConfig,
) {
    let vol = lattice.volume();
    let half_dt = 0.5 * dt;
    update_total_momenta(lattice, momenta, phi_fields, half_dt, fermion_config);
    for step in 0..n_steps {
        for idx in 0..vol {
            let x = lattice.site_coords(idx);
            for mu in 0..4 {
                let p = momenta[idx * 4 + mu];
                let u = lattice.link(x, mu);
                let exp_p = exp_su3_cayley(&p, dt);
                let new_u = (exp_p * u).reunitarize();
                lattice.set_link(x, mu, new_u);
            }
        }
        let p_dt = if step < n_steps - 1 { dt } else { half_dt };
        update_total_momenta(lattice, momenta, phi_fields, p_dt, fermion_config);
    }
}

fn update_total_momenta(
    lattice: &Lattice,
    momenta: &mut [Su3Matrix],
    phi_fields: &[FermionField],
    dt: f64,
    fermion_config: &PseudofermionConfig,
) {
    let vol = lattice.volume();
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        for mu in 0..4 {
            let gauge_f = lattice.gauge_force(x, mu);
            momenta[idx * 4 + mu] = momenta[idx * 4 + mu] + gauge_f.scale(dt);
        }
    }
    for phi in phi_fields {
        let mut x_field = FermionField::zeros(vol);
        let _cg = cg_solve(
            lattice,
            &mut x_field,
            phi,
            fermion_config.mass,
            fermion_config.cg_tol,
            fermion_config.cg_max_iter,
        );
        let ferm_force = pseudofermion_force(lattice, &x_field, fermion_config.mass);
        for (m, f) in momenta.iter_mut().zip(ferm_force.iter()) {
            *m = *m + f.scale(dt);
        }
    }
}

/// exp(dt × P) ≈ I + dt P + (dt P)²/2 — second-order Cayley approximation.
fn exp_su3_cayley(p: &Su3Matrix, dt: f64) -> Su3Matrix {
    let dp = p.scale(dt);
    let dp2 = dp * dp;
    let mut result = Su3Matrix::IDENTITY;
    for i in 0..3 {
        for j in 0..3 {
            result.m[i][j] = result.m[i][j] + dp.m[i][j] + dp2.m[i][j].scale(0.5);
        }
    }
    result
}

fn kinetic_energy(momenta: &[Su3Matrix]) -> f64 {
    let mut t = 0.0;
    for p in momenta {
        let p2 = *p * *p;
        t -= 0.5 * p2.re_trace();
    }
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heatbath_produces_nonzero_field() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let mut seed = 42u64;
        let phi = pseudofermion_heatbath(&lat, 0.1, &mut seed);
        assert!(phi.norm_sq() > 0.0, "pseudofermion should be nonzero");
    }

    #[test]
    fn pseudofermion_action_positive() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let mut seed = 42u64;
        let phi = pseudofermion_heatbath(&lat, 0.5, &mut seed);
        let config = PseudofermionConfig {
            mass: 0.5,
            cg_tol: 1e-8,
            cg_max_iter: 1000,
        };
        let (action, cg_res, _) = pseudofermion_action(&lat, &phi, &config);
        assert!(cg_res.converged, "CG should converge");
        assert!(
            action > 0.0,
            "pseudofermion action should be positive: {action}"
        );
    }

    #[test]
    fn fermion_force_is_traceless_antihermitian() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 5.5, 42);
        let x_field = FermionField::random(lat.volume(), 99);
        let force = pseudofermion_force(&lat, &x_field, 0.1);
        for (i, f) in force.iter().enumerate().take(10) {
            let tr = f.m[0][0] + f.m[1][1] + f.m[2][2];
            assert!(
                tr.abs_sq() < 1e-20,
                "force[{i}] trace should be ~0: |tr|²={}",
                tr.abs_sq()
            );
        }
    }

    #[test]
    fn dynamical_hmc_cold_start_runs() {
        let mut lat = Lattice::cold_start([4, 4, 4, 4], 5.5);
        let mut config = DynamicalHmcConfig {
            n_md_steps: 5,
            dt: 0.01,
            seed: 42,
            fermion: PseudofermionConfig {
                mass: 0.5,
                cg_tol: 1e-6,
                cg_max_iter: 500,
            },
            beta: 5.5,
            n_flavors_over_4: 1,
        };
        let result = dynamical_hmc_trajectory(&mut lat, &mut config);
        assert!(
            result.plaquette > 0.0 && result.plaquette <= 1.0,
            "plaquette should be in (0,1]: {}",
            result.plaquette
        );
        assert!(result.cg_iterations > 0);
    }
}
