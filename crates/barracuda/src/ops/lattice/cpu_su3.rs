// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-side SU(3) matrix operations for lattice gauge theory.
//!
//! Absorbed from hotSpring v0.64 `lattice/su3.rs` (Feb 2026).
//! The WGSL shader constant lives in `su3.rs`; this module provides
//! the Rust-side reference implementation for CPU lattice operations.

use std::ops::{Add, Mul, Sub};

use super::cpu_complex::Complex64;

/// 3×3 complex matrix — SU(3) link variable (row-major).
#[derive(Clone, Copy, Debug)]
#[must_use]
pub struct Su3Matrix {
    /// 3×3 matrix elements (row-major)
    pub m: [[Complex64; 3]; 3],
}

impl Mul for Su3Matrix {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                let mut s = Complex64::ZERO;
                for k in 0..3 {
                    s += self.m[i][k] * rhs.m[k][j];
                }
                r.m[i][j] = s;
            }
        }
        r
    }
}

impl Add for Su3Matrix {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j] + rhs.m[i][j];
            }
        }
        r
    }
}

impl Sub for Su3Matrix {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j] - rhs.m[i][j];
            }
        }
        r
    }
}

impl Su3Matrix {
    /// Identity matrix.
    pub const IDENTITY: Self = Self {
        m: [
            [Complex64::ONE, Complex64::ZERO, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ZERO, Complex64::ONE],
        ],
    };

    /// Zero matrix.
    pub const ZERO: Self = Self {
        m: [[Complex64::ZERO; 3]; 3],
    };

    /// Hermitian adjoint U†.
    pub fn adjoint(self) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[j][i].conj();
            }
        }
        r
    }

    /// Trace: sum of diagonal elements.
    pub fn trace(self) -> Complex64 {
        self.m[0][0] + self.m[1][1] + self.m[2][2]
    }

    /// Real part of trace.
    #[must_use]
    pub fn re_trace(self) -> f64 {
        self.m[0][0].re + self.m[1][1].re + self.m[2][2].re
    }

    /// Scale by real scalar.
    pub fn scale(self, s: f64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j].scale(s);
            }
        }
        r
    }

    /// Scale by complex scalar.
    pub fn scale_complex(self, s: Complex64) -> Self {
        let mut r = Self::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] = self.m[i][j] * s;
            }
        }
        r
    }

    /// Frobenius norm squared.
    #[must_use]
    pub fn norm_sq(self) -> f64 {
        let mut s = 0.0;
        for i in 0..3 {
            for j in 0..3 {
                s += self.m[i][j].abs_sq();
            }
        }
        s
    }

    /// Project back onto SU(3) via modified Gram-Schmidt reunitarization.
    pub fn reunitarize(self) -> Self {
        let mut u = self;
        let n0 = row_norm(&u, 0);
        if n0 > super::constants::LATTICE_DIVISION_GUARD {
            let inv = 1.0 / n0;
            for j in 0..3 {
                u.m[0][j] = u.m[0][j].scale(inv);
            }
        }
        let dot01 = row_dot(&u, 0, 1);
        for j in 0..3 {
            u.m[1][j] -= u.m[0][j] * dot01;
        }
        let n1 = row_norm(&u, 1);
        if n1 > super::constants::LATTICE_DIVISION_GUARD {
            let inv = 1.0 / n1;
            for j in 0..3 {
                u.m[1][j] = u.m[1][j].scale(inv);
            }
        }
        u.m[2][0] = (u.m[0][1] * u.m[1][2] - u.m[0][2] * u.m[1][1]).conj();
        u.m[2][1] = (u.m[0][2] * u.m[1][0] - u.m[0][0] * u.m[1][2]).conj();
        u.m[2][2] = (u.m[0][0] * u.m[1][1] - u.m[0][1] * u.m[1][0]).conj();
        u
    }

    /// Generate a random SU(3) matrix near identity.
    pub fn random_near_identity(seed: &mut u64, epsilon: f64) -> Self {
        use super::constants::lcg_gaussian;

        let mut h = [[Complex64::ZERO; 3]; 3];
        let mut rand_gauss = || -> f64 { lcg_gaussian(seed) };

        let a3 = rand_gauss() * epsilon;
        let a8 = rand_gauss() * epsilon;
        h[0][0] = Complex64::new(a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h[1][1] = Complex64::new(-a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h[2][2] = Complex64::new(-2.0 * a8 / 3.0_f64.sqrt(), 0.0);

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let re = rand_gauss() * epsilon;
            let im = rand_gauss() * epsilon;
            h[i][j] = Complex64::new(re, im);
            h[j][i] = Complex64::new(re, -im);
        }

        let mut result = Self::IDENTITY;
        for (i, row) in result.m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell += Complex64::I * h[i][j];
            }
        }
        for (i, row) in result.m.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let h2_ij = (0..3).fold(Complex64::ZERO, |acc, k| acc + h[i][k] * h[k][j]);
                *cell -= h2_ij.scale(0.5);
            }
        }

        result.reunitarize()
    }

    /// Generate a random su(3) Lie algebra element (traceless anti-Hermitian).
    pub fn random_algebra(seed: &mut u64) -> Self {
        use super::constants::lcg_gaussian;

        let scale = std::f64::consts::FRAC_1_SQRT_2;
        let mut rand_gauss = || -> f64 { scale * lcg_gaussian(seed) };

        let mut h = Self::ZERO;
        let a3 = rand_gauss();
        let a8 = rand_gauss();
        h.m[0][0] = Complex64::new(a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h.m[1][1] = Complex64::new(-a3 + a8 / 3.0_f64.sqrt(), 0.0);
        h.m[2][2] = Complex64::new(-2.0 * a8 / 3.0_f64.sqrt(), 0.0);

        for (i, j) in [(0, 1), (0, 2), (1, 2)] {
            let re = rand_gauss();
            let im = rand_gauss();
            h.m[i][j] = Complex64::new(re, im);
            h.m[j][i] = Complex64::new(re, -im);
        }

        h.scale_complex(Complex64::I)
    }
}

fn row_norm(u: &Su3Matrix, row: usize) -> f64 {
    let mut s = 0.0;
    for j in 0..3 {
        s += u.m[row][j].abs_sq();
    }
    s.sqrt()
}

fn row_dot(u: &Su3Matrix, r1: usize, r2: usize) -> Complex64 {
    let mut s = Complex64::ZERO;
    for j in 0..3 {
        s += u.m[r1][j].conj() * u.m[r2][j];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_properties() {
        let i = Su3Matrix::IDENTITY;
        assert!((i.re_trace() - 3.0).abs() < 1e-14);
    }

    #[test]
    fn unitarity_check() {
        let mut seed = 123u64;
        let u = Su3Matrix::random_near_identity(&mut seed, 0.2);
        let ud = u.adjoint();
        let prod = u * ud;
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (prod.m[i][j].re - expected).abs() < 1e-6,
                    "U U† not identity at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn reunitarize_fixes_drift() {
        let mut seed = 999u64;
        let mut u = Su3Matrix::random_near_identity(&mut seed, 0.5);
        u.m[0][0].re += 0.1;
        let fixed = u.reunitarize();
        let prod = fixed * fixed.adjoint();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((prod.m[i][j].re - expected).abs() < 1e-10);
            }
        }
    }
}
