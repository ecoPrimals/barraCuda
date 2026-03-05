// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU reference implementation of the staggered Dirac operator + CG solver.
//!
//! Absorbed from hotSpring v0.64 `lattice/dirac.rs` and `lattice/cg.rs`.
//! Used as the test oracle for the GPU `StaggeredDirac` pipeline and as
//! the fermion solver inside [`super::pseudofermion`].
//!
//! The GPU pipeline lives in `dirac.rs`; this module provides the CPU
//! reference that runs without a GPU device.

use super::cpu_complex::Complex64;
use super::cpu_su3::Su3Matrix;
use super::wilson::Lattice;

/// Color vector at a single lattice site: 3 complex components.
pub type ColorVector = [Complex64; 3];

/// Staggered fermion field: one `ColorVector` per lattice site.
pub struct FermionField {
    /// Color vectors at each site
    pub data: Vec<ColorVector>,
    /// Lattice volume (number of sites)
    pub volume: usize,
}

impl FermionField {
    /// Create a zero fermion field.
    pub fn zeros(volume: usize) -> Self {
        Self {
            data: vec![[Complex64::ZERO; 3]; volume],
            volume,
        }
    }

    /// Create a random fermion field (for testing).
    pub fn random(volume: usize, seed: u64) -> Self {
        use super::constants::lcg_uniform_f64;
        let mut rng = seed;
        let mut data = vec![[Complex64::ZERO; 3]; volume];
        for site in &mut data {
            for c in site.iter_mut() {
                let re = lcg_uniform_f64(&mut rng) - 0.5;
                let im = lcg_uniform_f64(&mut rng) - 0.5;
                *c = Complex64::new(re, im);
            }
        }
        Self { data, volume }
    }

    /// Inner product: <self|other>.
    pub fn dot(&self, other: &Self) -> Complex64 {
        let mut sum = Complex64::ZERO;
        for (a, b) in self.data.iter().zip(other.data.iter()) {
            for c in 0..3 {
                sum += a[c].conj() * b[c];
            }
        }
        sum
    }

    /// Squared norm: Re<self|self>.
    pub fn norm_sq(&self) -> f64 {
        self.dot(self).re
    }

    /// In-place: self += a * x.
    pub fn axpy(&mut self, a: Complex64, x: &Self) {
        for (si, xi) in self.data.iter_mut().zip(x.data.iter()) {
            for c in 0..3 {
                si[c] += a * xi[c];
            }
        }
    }

    /// Copy data from another field.
    pub fn copy_from(&mut self, other: &Self) {
        self.data.copy_from_slice(&other.data);
    }
}

fn staggered_phase(x: [usize; 4], mu: usize) -> f64 {
    let sum: usize = x.iter().take(mu).sum();
    if sum.is_multiple_of(2) {
        1.0
    } else {
        -1.0
    }
}

fn su3_times_vec(u: &Su3Matrix, v: &ColorVector) -> ColorVector {
    let mut result = [Complex64::ZERO; 3];
    for (c, r) in result.iter_mut().enumerate() {
        for (cp, vcp) in v.iter().enumerate() {
            *r += u.m[c][cp] * *vcp;
        }
    }
    result
}

fn su3_dagger_times_vec(u: &Su3Matrix, v: &ColorVector) -> ColorVector {
    let mut result = [Complex64::ZERO; 3];
    for (c, r) in result.iter_mut().enumerate() {
        for (cp, vcp) in v.iter().enumerate() {
            *r += u.m[cp][c].conj() * *vcp;
        }
    }
    result
}

/// Apply staggered Dirac operator: out = D_st × psi
pub fn apply_dirac(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let vol = lattice.volume();
    let mut result = FermionField::zeros(vol);
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let mut out = [Complex64::ZERO; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = psi.data[idx][c].scale(mass);
        }
        for mu in 0..4 {
            let eta = staggered_phase(x, mu);
            let half_eta = 0.5 * eta;
            let x_fwd = lattice.neighbor(x, mu, true);
            let idx_fwd = lattice.site_index(x_fwd);
            let u_fwd = lattice.link(x, mu);
            let fwd = su3_times_vec(&u_fwd, &psi.data[idx_fwd]);
            let x_bwd = lattice.neighbor(x, mu, false);
            let idx_bwd = lattice.site_index(x_bwd);
            let u_bwd = lattice.link(x_bwd, mu);
            let bwd = su3_dagger_times_vec(&u_bwd, &psi.data[idx_bwd]);
            for c in 0..3 {
                out[c] += (fwd[c] - bwd[c]).scale(half_eta);
            }
        }
        result.data[idx] = out;
    }
    result
}

/// Apply D† (adjoint of staggered Dirac operator).
pub fn apply_dirac_adjoint(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let vol = lattice.volume();
    let mut result = FermionField::zeros(vol);
    for idx in 0..vol {
        let x = lattice.site_coords(idx);
        let mut out = [Complex64::ZERO; 3];
        for (c, o) in out.iter_mut().enumerate() {
            *o = psi.data[idx][c].scale(mass);
        }
        for mu in 0..4 {
            let eta = staggered_phase(x, mu);
            let half_eta = 0.5 * eta;
            let x_fwd = lattice.neighbor(x, mu, true);
            let idx_fwd = lattice.site_index(x_fwd);
            let u_fwd = lattice.link(x, mu);
            let fwd = su3_times_vec(&u_fwd, &psi.data[idx_fwd]);
            let x_bwd = lattice.neighbor(x, mu, false);
            let idx_bwd = lattice.site_index(x_bwd);
            let u_bwd = lattice.link(x_bwd, mu);
            let bwd = su3_dagger_times_vec(&u_bwd, &psi.data[idx_bwd]);
            for c in 0..3 {
                out[c] -= (fwd[c] - bwd[c]).scale(half_eta);
            }
        }
        result.data[idx] = out;
    }
    result
}

/// Apply D†D (positive definite, for CG).
pub fn apply_dirac_sq(lattice: &Lattice, psi: &FermionField, mass: f64) -> FermionField {
    let dpsi = apply_dirac(lattice, psi, mass);
    apply_dirac_adjoint(lattice, &dpsi, mass)
}

// ─── CG solver ──────────────────────────────────────────────────────────────

/// CG solver result.
#[derive(Clone, Debug)]
pub struct CgResult {
    /// Whether the solver converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub final_residual: f64,
    /// Initial residual norm
    pub initial_residual: f64,
}

/// Solve D†D x = b using Conjugate Gradient.
pub fn cg_solve(
    lattice: &Lattice,
    x: &mut FermionField,
    b: &FermionField,
    mass: f64,
    tol: f64,
    max_iter: usize,
) -> CgResult {
    let vol = lattice.volume();
    let ax = apply_dirac_sq(lattice, x, mass);
    let mut r = FermionField::zeros(vol);
    for i in 0..vol {
        for c in 0..3 {
            r.data[i][c] = b.data[i][c] - ax.data[i][c];
        }
    }
    let b_norm_sq = b.norm_sq();
    if b_norm_sq < super::constants::LATTICE_DIVISION_GUARD {
        return CgResult {
            converged: true,
            iterations: 0,
            final_residual: 0.0,
            initial_residual: 0.0,
        };
    }
    let mut r_norm_sq = r.norm_sq();
    let initial_residual = (r_norm_sq / b_norm_sq).sqrt();
    let tol_sq = tol * tol * b_norm_sq;
    if r_norm_sq < tol_sq {
        return CgResult {
            converged: true,
            iterations: 0,
            final_residual: initial_residual,
            initial_residual,
        };
    }
    let mut p = FermionField::zeros(vol);
    p.copy_from(&r);
    let mut iterations = 0;
    for iter in 0..max_iter {
        iterations = iter + 1;
        let ap = apply_dirac_sq(lattice, &p, mass);
        let p_ap = p.dot(&ap).re;
        if p_ap.abs() < super::constants::LATTICE_DIVISION_GUARD {
            break;
        }
        let alpha = r_norm_sq / p_ap;
        x.axpy(Complex64::new(alpha, 0.0), &p);
        r.axpy(Complex64::new(-alpha, 0.0), &ap);
        let r_norm_sq_new = r.norm_sq();
        if r_norm_sq_new < tol_sq {
            r_norm_sq = r_norm_sq_new;
            break;
        }
        let beta = r_norm_sq_new / r_norm_sq;
        r_norm_sq = r_norm_sq_new;
        for i in 0..vol {
            for c in 0..3 {
                p.data[i][c] = r.data[i][c] + p.data[i][c].scale(beta);
            }
        }
    }
    let final_residual = (r_norm_sq / b_norm_sq).sqrt();
    CgResult {
        converged: final_residual < tol,
        iterations,
        final_residual,
        initial_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dirac_on_zero_field_is_zero() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let psi = FermionField::zeros(lat.volume());
        let result = apply_dirac(&lat, &psi, 0.1);
        assert!(result.norm_sq() < 1e-20);
    }

    #[test]
    fn dirac_sq_positive_definite() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let psi = FermionField::random(lat.volume(), 99);
        let ddpsi = apply_dirac_sq(&lat, &psi, 0.1);
        let inner = psi.dot(&ddpsi).re;
        assert!(inner > 0.0, "<ψ|D†D|ψ> should be positive: {inner}");
    }

    #[test]
    fn cg_identity_lattice() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = FermionField::random(vol, 42);
        let mut x = FermionField::zeros(vol);
        let result = cg_solve(&lat, &mut x, &b, 1.0, 1e-8, 500);
        assert!(
            result.converged,
            "CG should converge: r={}",
            result.final_residual
        );
    }

    #[test]
    fn cg_hot_lattice() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let vol = lat.volume();
        let b = FermionField::random(vol, 123);
        let mut x = FermionField::zeros(vol);
        let result = cg_solve(&lat, &mut x, &b, 0.5, 1e-4, 2000);
        assert!(
            result.converged,
            "CG should converge on hot lattice: r={}, iters={}",
            result.final_residual, result.iterations
        );
    }
}
