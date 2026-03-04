// SPDX-License-Identifier: AGPL-3.0-or-later

//! CPU-side Complex64 arithmetic for lattice field theory.
//!
//! Absorbed from hotSpring v0.64 `lattice/complex_f64.rs` (Feb 2026).
//! The WGSL shader constant lives in `complex_f64.rs`; this module provides
//! the Rust-side reference implementation for CPU lattice operations.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Complex number with f64 real and imaginary parts.
#[derive(Clone, Copy, Debug, PartialEq)]
#[must_use]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}

impl Complex64 {
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };
    pub const I: Self = Self { re: 0.0, im: 1.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    #[inline]
    pub fn abs_sq(self) -> f64 {
        self.re.mul_add(self.re, self.im * self.im)
    }

    #[inline]
    pub fn abs(self) -> f64 {
        self.abs_sq().sqrt()
    }

    #[inline]
    pub fn exp(self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    #[inline]
    pub fn scale(self, s: f64) -> Self {
        Self {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

impl Add for Complex64 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl AddAssign for Complex64 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.re += rhs.re;
        self.im += rhs.im;
    }
}

impl Sub for Complex64 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl SubAssign for Complex64 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.re -= rhs.re;
        self.im -= rhs.im;
    }
}

impl Mul for Complex64 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re.mul_add(rhs.re, -(self.im * rhs.im)),
            im: self.re.mul_add(rhs.im, self.im * rhs.re),
        }
    }
}

impl MulAssign for Complex64 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Complex64 {
    type Output = Self;
    #[inline]
    #[expect(clippy::suspicious_arithmetic_impl, reason = "suppressed")]
    fn div(self, rhs: Self) -> Self {
        let d = rhs.abs_sq();
        Self {
            re: self.re.mul_add(rhs.re, self.im * rhs.im) / d,
            im: self.im.mul_add(rhs.re, -(self.re * rhs.im)) / d,
        }
    }
}

impl Neg for Complex64 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl fmt::Display for Complex64 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.6}+{:.6}i", self.re, self.im)
        } else {
            write!(f, "{:.6}{:.6}i", self.re, self.im)
        }
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn complex_basic_ops() {
        let a = Complex64::new(1.0, 2.0);
        let b = Complex64::new(3.0, -1.0);
        let c = a + b;
        assert!((c.re - 4.0).abs() < 1e-15);
        assert!((c.im - 1.0).abs() < 1e-15);
        let d = a * b;
        assert!((d.re - 5.0).abs() < 1e-15);
        assert!((d.im - 5.0).abs() < 1e-15);
    }

    #[test]
    fn complex_conj_abs() {
        let a = Complex64::new(3.0, 4.0);
        assert!((a.abs() - 5.0).abs() < 1e-15);
        let c = a.conj();
        assert!((c.im - (-4.0)).abs() < 1e-15);
    }
}
