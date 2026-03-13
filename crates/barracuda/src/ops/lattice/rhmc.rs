// SPDX-License-Identifier: AGPL-3.0-only

//! Rational Hybrid Monte Carlo (RHMC) — rational approximation and multi-shift CG.
//!
//! Enables simulation of non-multiples-of-4 flavor counts (Nf=2, 2+1)
//! via the "rooting trick": det(D†D)^{Nf/8} ≈ product of rational functions.
//!
//! # Components
//!
//! - [`RationalApproximation`]: Partial fraction coefficients for x^{p/q}
//!   (production code — pure math, no lattice dependencies)
//! - [`multi_shift_cg_solve`]: Solves (D†D + `σ_i`)`x_i` = b for all shifts
//!   simultaneously with a single Krylov space (test-only CPU reference)
//!
//! Absorbed from hotSpring v0.64 `lattice/rhmc.rs` (Mar 2026).
//!
//! # References
//!
//! - Clark & Kennedy, NPB 552 (1999) 461 — RHMC algorithm
//! - Clark, "The Rational Hybrid Monte Carlo Algorithm" (2006)
//! - Remez algorithm for optimal rational approximation

/// Partial-fraction representation of a rational approximation:
///   r(x) = `alpha_0` + Σ_{i=1}^{n} `alpha_i` / (x + `sigma_i`)
///
/// Used to approximate x^{p/q} over [`lambda_min`, `lambda_max`].
#[derive(Clone, Debug)]
pub struct RationalApproximation {
    /// Constant term.
    pub alpha_0: f64,
    /// Residues (numerator coefficients).
    pub alpha: Vec<f64>,
    /// Shifts (denominator offsets, all positive).
    pub sigma: Vec<f64>,
    /// Power being approximated (e.g. 0.25 for x^{1/4}).
    pub power: f64,
    /// Spectral range lower bound.
    pub lambda_min: f64,
    /// Spectral range upper bound.
    pub lambda_max: f64,
    /// Maximum relative error over the spectral range.
    pub max_relative_error: f64,
}

impl RationalApproximation {
    /// Number of poles (shifts) in the approximation.
    #[must_use]
    pub fn n_poles(&self) -> usize {
        self.alpha.len()
    }

    /// Evaluate the rational approximation at point x.
    #[must_use]
    pub fn eval(&self, x: f64) -> f64 {
        let mut val = self.alpha_0;
        for (a, s) in self.alpha.iter().zip(&self.sigma) {
            val += a / (x + s);
        }
        val
    }

    /// Generate an n-pole rational approximation to x^power on [`lambda_min`, `lambda_max`].
    ///
    /// Uses geometric pole initialization, Remez exchange for residue fitting,
    /// and coordinate descent for pole optimization.
    #[must_use]
    pub fn generate(power: f64, n_poles: usize, lambda_min: f64, lambda_max: f64) -> Self {
        let log_min = lambda_min.ln();
        let log_max = lambda_max.ln();

        let mut sigma: Vec<f64> = (0..n_poles)
            .map(|i| {
                let t = (i as f64 + 0.5) / n_poles as f64;
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect();

        let n_eval = 4000;
        let eval_grid: Vec<f64> = (0..n_eval)
            .map(|i| {
                let t = f64::from(i) / f64::from(n_eval - 1);
                (log_min + t * (log_max - log_min)).exp()
            })
            .collect();

        let (mut best_coeffs, mut best_err) = remez_for_poles(&sigma, power, &eval_grid);

        for _ in 0..40 {
            let mut improved = false;
            for i in 0..n_poles {
                let log_s = sigma[i].ln();
                let lower = if i > 0 {
                    sigma[i - 1].ln() + 0.01
                } else {
                    log_min - 3.0
                };
                let upper = if i + 1 < n_poles {
                    sigma[i + 1].ln() - 0.01
                } else {
                    log_max + 3.0
                };

                let gr = 0.5 * (5.0_f64.sqrt() - 1.0);
                let mut a = lower;
                let mut b = upper;
                let mut c = b - gr * (b - a);
                let mut d = a + gr * (b - a);

                sigma[i] = c.exp();
                let (_, mut fc) = remez_for_poles(&sigma, power, &eval_grid);
                sigma[i] = d.exp();
                let (_, mut fd) = remez_for_poles(&sigma, power, &eval_grid);

                for _ in 0..25 {
                    if (b - a).abs() < 0.005 {
                        break;
                    }
                    if fc < fd {
                        b = d;
                        d = c;
                        fd = fc;
                        c = b - gr * (b - a);
                        sigma[i] = c.exp();
                        let (_, e) = remez_for_poles(&sigma, power, &eval_grid);
                        fc = e;
                    } else {
                        a = c;
                        c = d;
                        fc = fd;
                        d = a + gr * (b - a);
                        sigma[i] = d.exp();
                        let (_, e) = remez_for_poles(&sigma, power, &eval_grid);
                        fd = e;
                    }
                }

                let (best_pos, best_pos_err) = if fc < fd { (c, fc) } else { (d, fd) };

                sigma[i] = log_s.exp();
                let (_, cur_err) = remez_for_poles(&sigma, power, &eval_grid);
                if best_pos_err < cur_err * 0.998 {
                    sigma[i] = best_pos.exp();
                    improved = true;
                }
            }

            let (c, e) = remez_for_poles(&sigma, power, &eval_grid);
            if e < best_err {
                best_err = e;
                best_coeffs = c;
            }
            if !improved {
                break;
            }
        }

        Self {
            alpha_0: best_coeffs[0],
            alpha: best_coeffs[1..].to_vec(),
            sigma,
            power,
            lambda_min,
            lambda_max,
            max_relative_error: best_err,
        }
    }

    /// 8-pole approximation to x^{1/4} on \[0.01, 64\].
    #[must_use]
    pub fn fourth_root_8pole() -> Self {
        Self::generate(0.25, 8, 0.01, 64.0)
    }

    /// 8-pole approximation to x^{-1/4} on \[0.01, 64\].
    #[must_use]
    pub fn inv_fourth_root_8pole() -> Self {
        Self::generate(-0.25, 8, 0.01, 64.0)
    }

    /// 8-pole approximation to x^{1/2} on \[0.01, 64\].
    #[must_use]
    pub fn sqrt_8pole() -> Self {
        Self::generate(0.5, 8, 0.01, 64.0)
    }

    /// 8-pole approximation to x^{-1/2} on \[0.01, 64\].
    #[must_use]
    pub fn inv_sqrt_8pole() -> Self {
        Self::generate(-0.5, 8, 0.01, 64.0)
    }
}

/// Result of a multi-shift CG solve.
#[derive(Clone, Debug)]
pub struct MultiShiftCgResult {
    /// Number of CG iterations (shared across all shifts).
    pub iterations: usize,
    /// Final residual norm squared (for the base system, sigma=0).
    pub residual_sq: f64,
    /// Whether the solver converged.
    pub converged: bool,
}

/// Multi-shift Conjugate Gradient: solve (D†D + `σ_i`)`x_i` = b for all shifts simultaneously.
///
/// All shifted systems share the same Krylov space. Only one matrix-vector
/// product (D†D·p) per iteration, regardless of the number of shifts.
///
/// Returns solution vectors `x_i` (one per shift) and convergence info.
#[cfg(test)]
#[must_use]
pub fn multi_shift_cg_solve(
    lattice: &super::wilson::Lattice,
    b: &super::cpu_dirac::FermionField,
    mass: f64,
    shifts: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<super::cpu_dirac::FermionField>, MultiShiftCgResult) {
    use super::cpu_dirac::{FermionField, apply_dirac_sq};

    let vol = lattice.volume();
    let n_shifts = shifts.len();

    let mut x: Vec<FermionField> = (0..n_shifts).map(|_| FermionField::zeros(vol)).collect();
    let mut p: Vec<FermionField> = (0..n_shifts)
        .map(|_| FermionField {
            data: b.data.clone(),
            volume: vol,
        })
        .collect();
    let mut r = FermionField {
        data: b.data.clone(),
        volume: vol,
    };

    let b_norm_sq = b.dot(b).re;
    if b_norm_sq < 1e-30 {
        return (
            x,
            MultiShiftCgResult {
                iterations: 0,
                residual_sq: 0.0,
                converged: true,
            },
        );
    }

    let tol_sq = tol * tol * b_norm_sq;
    let mut rz = b_norm_sq;
    let mut zeta_prev: Vec<f64> = vec![1.0; n_shifts];
    let mut zeta_curr: Vec<f64> = vec![1.0; n_shifts];
    let mut beta_prev: Vec<f64> = vec![0.0; n_shifts];
    let mut alpha_prev = 0.0_f64;
    let mut active: Vec<bool> = vec![true; n_shifts];

    let mut iterations = 0;

    for _iter in 0..max_iter {
        iterations += 1;

        let ap = apply_dirac_sq(lattice, &p[0], mass);

        let mut p_ap = p[0].dot(&ap).re;
        p_ap += shifts[0] * p[0].dot(&p[0]).re;

        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;

        for i in 0..vol {
            for c in 0..3 {
                x[0].data[i][c] += p[0].data[i][c].scale(alpha);
                r.data[i][c] -= (ap.data[i][c] + p[0].data[i][c].scale(shifts[0])).scale(alpha);
            }
        }

        let rz_new = r.dot(&r).re;

        for s in 1..n_shifts {
            if !active[s] {
                continue;
            }

            let ds = shifts[s] - shifts[0];
            let denom = alpha.mul_add(ds, 1.0)
                + alpha * alpha_prev * (1.0 - zeta_prev[s] / zeta_curr[s])
                    / beta_prev[s].max(1e-30);
            if denom.abs() < 1e-30 {
                active[s] = false;
                continue;
            }

            let zeta_next = zeta_curr[s] / denom;
            let alpha_s = alpha * zeta_next / zeta_curr[s];

            for i in 0..vol {
                for c in 0..3 {
                    x[s].data[i][c] += p[s].data[i][c].scale(alpha_s);
                }
            }

            let beta_s = if rz.abs() > 1e-30 {
                (zeta_next / zeta_curr[s]).powi(2) * (rz_new / rz)
            } else {
                0.0
            };

            for i in 0..vol {
                for c in 0..3 {
                    p[s].data[i][c] = r.data[i][c].scale(zeta_next) + p[s].data[i][c].scale(beta_s);
                }
            }

            zeta_prev[s] = zeta_curr[s];
            zeta_curr[s] = zeta_next;
            beta_prev[s] = beta_s;
        }

        let beta = if rz.abs() > 1e-30 { rz_new / rz } else { 0.0 };
        for i in 0..vol {
            for c in 0..3 {
                p[0].data[i][c] = r.data[i][c] + p[0].data[i][c].scale(beta);
            }
        }

        alpha_prev = alpha;
        rz = rz_new;

        if rz < tol_sq {
            break;
        }
    }

    (
        x,
        MultiShiftCgResult {
            iterations,
            residual_sq: rz / b_norm_sq,
            converged: rz < tol_sq,
        },
    )
}

/// Remez exchange algorithm for fixed poles: finds residues that minimize max relative error.
fn remez_for_poles(sigma: &[f64], power: f64, eval_grid: &[f64]) -> (Vec<f64>, f64) {
    let n_poles = sigma.len();
    let ncols = n_poles + 1;
    let n_sys = ncols + 1;

    let n_eval = eval_grid.len();
    let mut ref_indices: Vec<usize> = (0..n_sys)
        .map(|k| {
            let theta = std::f64::consts::PI * k as f64 / (n_sys - 1) as f64;
            let t = 0.5 * (1.0 - theta.cos());
            ((t * (n_eval - 1) as f64) as usize).min(n_eval - 1)
        })
        .collect();
    ref_indices.sort_unstable();
    ref_indices.dedup();
    while ref_indices.len() < n_sys {
        for idx in 0..n_eval {
            if !ref_indices.contains(&idx) {
                ref_indices.push(idx);
                ref_indices.sort_unstable();
                break;
            }
        }
    }

    let mut best_coeffs = vec![0.0_f64; ncols];
    let mut best_err = f64::MAX;

    for _ in 0..60 {
        let mut mat = vec![0.0_f64; n_sys * n_sys];
        let rhs = vec![1.0_f64; n_sys];
        for k in 0..n_sys {
            let x = eval_grid[ref_indices[k]];
            let t = x.powf(power);
            mat[k * n_sys] = 1.0 / t;
            for i in 0..n_poles {
                mat[k * n_sys + i + 1] = 1.0 / ((x + sigma[i]) * t);
            }
            mat[k * n_sys + ncols] = if k % 2 == 0 { -1.0 } else { 1.0 };
        }

        let solution = solve_linear_system(&mat, &rhs, n_sys);
        let coeffs: Vec<f64> = solution[..ncols].to_vec();

        let mut max_abs = 0.0_f64;
        let mut signed_err = Vec::with_capacity(n_eval);
        for (idx, &x) in eval_grid.iter().enumerate() {
            let exact = x.powf(power);
            let mut val = coeffs[0];
            for (a, s) in coeffs[1..].iter().zip(sigma) {
                val += a / (x + s);
            }
            let se = (val - exact) / exact;
            signed_err.push((idx, se));
            if se.abs() > max_abs {
                max_abs = se.abs();
            }
        }

        if max_abs < best_err {
            best_err = max_abs;
            best_coeffs = coeffs;
        }
        if max_abs < 1e-12 {
            break;
        }

        let mut extrema: Vec<(usize, f64)> = Vec::new();
        extrema.push(signed_err[0]);
        for i in 1..n_eval - 1 {
            let (_, ep) = signed_err[i - 1];
            let (idx, e) = signed_err[i];
            let (_, en) = signed_err[i + 1];
            if (e > ep && e > en) || (e < ep && e < en) {
                extrema.push((idx, e));
            }
        }
        extrema.push(signed_err[n_eval - 1]);

        if extrema.len() < n_sys {
            break;
        }

        let mut selected: Vec<(usize, f64)> = Vec::new();
        for &(idx, e) in &extrema {
            if selected.is_empty() {
                selected.push((idx, e));
            } else {
                let Some(last) = selected.last() else {
                    continue;
                };
                let last_sign = last.1.signum();
                if (e.signum() - last_sign).abs() > f64::EPSILON {
                    selected.push((idx, e));
                } else if selected.last().is_some_and(|l| e.abs() > l.1.abs()) {
                    if let Some(slot) = selected.last_mut() {
                        *slot = (idx, e);
                    }
                }
            }
        }

        while selected.len() > n_sys {
            let Some((min_idx, _)) = selected
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.1.abs().total_cmp(&b.1.abs()))
            else {
                break;
            };
            selected.remove(min_idx);
        }

        if selected.len() < n_sys {
            break;
        }

        let new_refs: Vec<usize> = selected.iter().map(|&(idx, _)| idx).collect();
        if new_refs == ref_indices {
            break;
        }
        ref_indices = new_refs;
    }

    (best_coeffs, best_err)
}

/// Solve A·x = b via Gaussian elimination with partial pivoting.
fn solve_linear_system(ata: &[f64], atb: &[f64], n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    a.copy_from_slice(&ata[..n * n]);
    let mut b = atb[..n].to_vec();

    for col in 0..n {
        let mut max_row = col;
        let mut max_val = a[col * n + col].abs();
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }
        let pivot = a[col * n + col];
        if pivot.abs() < 1e-30 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = b[i];
        for j in (i + 1)..n {
            sum -= a[i * n + j] * x[j];
        }
        let diag = a[i * n + i];
        x[i] = if diag.abs() > 1e-30 { sum / diag } else { 0.0 };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_approx_evaluates_correctly() {
        let r = RationalApproximation::fourth_root_8pole();
        let approx_val = r.eval(1.0);
        assert!(
            (approx_val - 1.0).abs() < 0.01,
            "fourth_root(1.0) = {approx_val}, expected ~1.0"
        );

        let approx_val = r.eval(16.0);
        assert!(
            (approx_val - 2.0).abs() < 0.01,
            "fourth_root(16.0) = {approx_val}, expected ~2.0"
        );
    }

    #[test]
    fn inv_fourth_root_evaluates_correctly() {
        let r = RationalApproximation::inv_fourth_root_8pole();
        let approx_val = r.eval(1.0);
        assert!(
            (approx_val - 1.0).abs() < 0.01,
            "inv_fourth_root(1.0) = {approx_val}, expected ~1.0"
        );
    }

    #[test]
    fn sqrt_evaluates_correctly() {
        let r = RationalApproximation::sqrt_8pole();
        let approx_val = r.eval(4.0);
        let rel_err = (approx_val - 2.0).abs() / 2.0;
        assert!(
            rel_err < 0.02,
            "sqrt(4.0) = {approx_val}, expected ~2.0, rel_err={rel_err:.4}"
        );
        assert!(
            r.max_relative_error < 0.05,
            "max relative error {} too large for 8-pole sqrt",
            r.max_relative_error
        );
    }

    #[test]
    fn multi_shift_cg_solves_base_system() {
        let lat = super::super::wilson::Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = super::super::cpu_dirac::FermionField::random(vol, 42);
        let shifts = vec![0.0, 0.1, 1.0];
        let (solutions, result) = multi_shift_cg_solve(&lat, &b, 1.0, &shifts, 1e-6, 500);
        assert_eq!(solutions.len(), 3);
        assert!(
            result.converged,
            "multi-shift CG should converge on cold config"
        );
        assert!(
            result.residual_sq < 1e-10,
            "residual too large: {}",
            result.residual_sq
        );
    }

    #[test]
    fn multi_shift_cg_zero_rhs() {
        let lat = super::super::wilson::Lattice::cold_start([4, 4, 4, 4], 6.0);
        let vol = lat.volume();
        let b = super::super::cpu_dirac::FermionField::zeros(vol);
        let shifts = vec![0.0, 0.5];
        let (_, result) = multi_shift_cg_solve(&lat, &b, 1.0, &shifts, 1e-8, 100);
        assert!(result.converged);
        assert_eq!(result.iterations, 0);
    }
}
