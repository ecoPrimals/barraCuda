// SPDX-License-Identifier: AGPL-3.0-or-later

//! Wilson gauge action for SU(3) lattice gauge theory.
//!
//! The fundamental building block is the plaquette — the smallest closed loop
//! of link variables on the lattice:
//!
//!   `P_μν(x)` = `U_μ(x)` `U_ν(x+μ)` `U_μ†(x+ν)` `U_ν†(x)`
//!
//! The Wilson action is:
//!
//!   S = β × Σ_{x,μ<ν} (1 - Re Tr `P_μν(x)` / 3)
//!
//! where β = 6/g² is the inverse bare coupling.
//!
//! Absorbed from hotSpring v0.64 `lattice/wilson.rs` (Feb 2026).
//!
//! # References
//!
//! - Wilson, PRD 10, 2445 (1974)
//! - Gattringer & Lang, "QCD on the Lattice" (2010), Ch. 3

use super::cpu_complex::Complex64;
use super::cpu_su3::Su3Matrix;

/// 4D lattice of SU(3) link variables.
///
/// Links are stored as `links[site * 4 + mu]` where mu ∈ {0,1,2,3}
/// represents the four spacetime directions.
pub struct Lattice {
    /// Lattice dimensions [nx, ny, nz, nt]
    pub dims: [usize; 4],
    /// SU(3) link variables (volume × 4)
    pub links: Vec<Su3Matrix>,
    /// Inverse bare coupling β = 6/g²
    pub beta: f64,
}

impl Lattice {
    /// Total lattice volume (nx × ny × nz × nt).
    #[must_use]
    pub const fn volume(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2] * self.dims[3]
    }

    /// Linear index from 4D coordinates.
    #[must_use]
    pub const fn site_index(&self, x: [usize; 4]) -> usize {
        x[0] + self.dims[0] * (x[1] + self.dims[1] * (x[2] + self.dims[2] * x[3]))
    }

    /// 4D coordinates from linear index.
    #[must_use]
    pub const fn site_coords(&self, idx: usize) -> [usize; 4] {
        let x0 = idx % self.dims[0];
        let rem = idx / self.dims[0];
        let x1 = rem % self.dims[1];
        let rem2 = rem / self.dims[1];
        let x2 = rem2 % self.dims[2];
        let x3 = rem2 / self.dims[2];
        [x0, x1, x2, x3]
    }

    /// Neighbor site in direction μ (forward or backward with PBC).
    #[must_use]
    pub const fn neighbor(&self, x: [usize; 4], mu: usize, forward: bool) -> [usize; 4] {
        let mut y = x;
        if forward {
            y[mu] = (x[mu] + 1) % self.dims[mu];
        } else {
            y[mu] = (x[mu] + self.dims[mu] - 1) % self.dims[mu];
        }
        y
    }

    /// Get link variable at site x in direction μ.
    pub fn link(&self, x: [usize; 4], mu: usize) -> Su3Matrix {
        let idx = self.site_index(x);
        self.links[idx * 4 + mu]
    }

    /// Set link variable at site x in direction μ.
    pub fn set_link(&mut self, x: [usize; 4], mu: usize, u: Su3Matrix) {
        let idx = self.site_index(x);
        self.links[idx * 4 + mu] = u;
    }

    /// Cold start: all links = identity.
    #[must_use]
    pub fn cold_start(dims: [usize; 4], beta: f64) -> Self {
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        Self {
            dims,
            links: vec![Su3Matrix::IDENTITY; vol * 4],
            beta,
        }
    }

    /// Hot start: random links near identity.
    #[must_use]
    pub fn hot_start(dims: [usize; 4], beta: f64, seed: u64) -> Self {
        let vol = dims[0] * dims[1] * dims[2] * dims[3];
        let mut rng_seed = seed;
        let links: Vec<Su3Matrix> = (0..vol * 4)
            .map(|_| Su3Matrix::random_near_identity(&mut rng_seed, 1.5))
            .collect();
        Self { dims, links, beta }
    }

    /// Plaquette `P_μν(x)` = `U_μ(x)` `U_ν(x+μ)` `U_μ†(x+ν)` `U_ν†(x)`.
    pub fn plaquette(&self, x: [usize; 4], mu: usize, nu: usize) -> Su3Matrix {
        let x_mu = self.neighbor(x, mu, true);
        let x_nu = self.neighbor(x, nu, true);
        let u1 = self.link(x, mu);
        let u2 = self.link(x_mu, nu);
        let u3 = self.link(x_nu, mu).adjoint();
        let u4 = self.link(x, nu).adjoint();
        u1 * u2 * u3 * u4
    }

    /// Volume-averaged plaquette ⟨Re Tr P / 3⟩.
    #[must_use]
    pub fn average_plaquette(&self) -> f64 {
        let vol = self.volume();
        let mut sum = 0.0;
        let mut count = 0usize;
        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self.plaquette(x, mu, nu);
                    sum += p.re_trace() / 3.0;
                    count += 1;
                }
            }
        }
        sum / count as f64
    }

    /// Staple sum for link (x, μ) used in gauge force.
    pub fn staple(&self, x: [usize; 4], mu: usize) -> Su3Matrix {
        let mut s = Su3Matrix::ZERO;
        let x_mu = self.neighbor(x, mu, true);
        for nu in 0..4 {
            if nu == mu {
                continue;
            }
            let x_nu = self.neighbor(x, nu, true);
            let x_mu_bnu = self.neighbor(x_mu, nu, false);
            let x_bnu = self.neighbor(x, nu, false);
            let upper =
                self.link(x_mu, nu) * self.link(x_nu, mu).adjoint() * self.link(x, nu).adjoint();
            let lower = self.link(x_mu_bnu, nu).adjoint()
                * self.link(x_bnu, mu).adjoint()
                * self.link(x_bnu, nu);
            s = s + upper + lower;
        }
        s
    }

    /// Wilson gauge action S = β Σ (1 - Re Tr P / 3).
    #[must_use]
    pub fn wilson_action(&self) -> f64 {
        let vol = self.volume();
        let mut sum = 0.0;
        for idx in 0..vol {
            let x = self.site_coords(idx);
            for mu in 0..4 {
                for nu in (mu + 1)..4 {
                    let p = self.plaquette(x, mu, nu);
                    sum += 1.0 - p.re_trace() / 3.0;
                }
            }
        }
        self.beta * sum
    }

    /// Gauge force `dS/dU_μ(x)` (traceless antihermitian).
    pub fn gauge_force(&self, x: [usize; 4], mu: usize) -> Su3Matrix {
        let u = self.link(x, mu);
        let v = self.staple(x, mu);
        let w = u * v;
        let wd = w.adjoint();
        let diff = (w - wd).scale(0.5);
        let tr = diff.trace();
        let tr_over_3 = tr.scale(1.0 / 3.0);
        let mut proj = diff;
        for i in 0..3 {
            proj.m[i][i] -= tr_over_3;
        }
        proj.scale(-self.beta / 3.0)
    }

    /// Polyakov loop: product of temporal links at a fixed spatial position.
    pub fn polyakov_loop(&self, x_spatial: [usize; 3]) -> Complex64 {
        let nt = self.dims[3];
        let mut prod = Su3Matrix::IDENTITY;
        for t in 0..nt {
            let x = [x_spatial[0], x_spatial[1], x_spatial[2], t];
            prod = prod * self.link(x, 3);
        }
        prod.trace().scale(1.0 / 3.0)
    }

    /// Volume-averaged Polyakov loop returning (Re, Im) components.
    ///
    /// Averages Tr P(x) / 3 over all spatial sites, returning both the real
    /// and imaginary parts. A nonzero ⟨Re P⟩ signals deconfinement; the
    /// imaginary part serves as a Z(3) symmetry diagnostic.
    #[must_use]
    pub fn complex_polyakov_average(&self) -> (f64, f64) {
        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        let spatial_vol = nx * ny * nz;
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let p = self.polyakov_loop([ix, iy, iz]);
                    sum_re += p.re;
                    sum_im += p.im;
                }
            }
        }
        let inv_vol = 1.0 / spatial_vol as f64;
        (sum_re * inv_vol, sum_im * inv_vol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_start_plaquette_is_one() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let p = lat.average_plaquette();
        assert!(
            (p - 1.0).abs() < 1e-14,
            "cold start plaquette should be 1.0, got {p}"
        );
    }

    #[test]
    fn cold_start_action_is_zero() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let s = lat.wilson_action();
        assert!(s.abs() < 1e-12, "cold start action should be 0, got {s}");
    }

    #[test]
    fn hot_start_plaquette_below_one() {
        let lat = Lattice::hot_start([4, 4, 4, 4], 6.0, 42);
        let p = lat.average_plaquette();
        assert!(p < 1.0, "hot start plaquette should be < 1.0, got {p}");
        assert!(p > -1.0, "plaquette should be > -1.0, got {p}");
    }

    #[test]
    fn site_index_roundtrip() {
        let lat = Lattice::cold_start([4, 6, 8, 10], 6.0);
        for idx in 0..lat.volume() {
            let coords = lat.site_coords(idx);
            let back = lat.site_index(coords);
            assert_eq!(idx, back, "site index roundtrip failed at {idx}");
        }
    }

    #[test]
    fn gauge_force_cold_start_is_zero() {
        let lat = Lattice::cold_start([4, 4, 4, 4], 6.0);
        let f = lat.gauge_force([1, 1, 1, 1], 0);
        assert!(f.norm_sq() < 1e-20, "force on cold start should be zero");
    }
}
