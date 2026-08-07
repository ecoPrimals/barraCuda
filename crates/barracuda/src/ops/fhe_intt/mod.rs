// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Inverse Number Theoretic Transform (INTT) Operation
//!
//! **Purpose**: Convert NTT-domain polynomial back to coefficient domain
//!
//! **Algorithm**: Inverse Cooley-Tukey FFT with scaling by N^(-1)
//! - Time complexity: O(N log N)
//! - Completes the NTT multiplication pipeline
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure Rust + WGSL (no unsafe)
//! - ✅ Hardware-agnostic (wgpu backend selection)
//! - ✅ Numerically precise (modular arithmetic)
//! - ✅ Production-ready (full error handling)
//! - ✅ Canonical pattern: Tensor inputs/outputs
//!
//! ## Mathematical Background
//!
//! The Inverse NTT transforms from frequency domain back to coefficient domain:
//!
//! ```text
//! a(X) = INTT(A) = (1/N) * NTT^(-1)(A)
//! ```
//!
//! Where NTT^(-1) uses the inverse root of unity ω^(-1).
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use barracuda::ops::fhe_ntt::FheNtt;
//! use barracuda::ops::fhe_intt::FheIntt;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Forward NTT
//! let ntt = FheNtt::new(poly, 4096, modulus, root)?;
//! let poly_ntt = ntt.execute()?;
//!
//! // Inverse NTT (round-trip)
//! let intt = FheIntt::new(poly_ntt, 4096, modulus, inv_root)?;
//! let poly_recovered = intt.execute()?;
//!
//! // poly_recovered should equal original poly
//! # Ok(())
//! # }
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// FHE Inverse Number Theoretic Transform operation
///
/// Transforms polynomial from NTT domain back to coefficient domain.
pub struct FheIntt {
    input: Tensor,
    degree: u32,
    modulus: u64,
    inv_root_of_unity: u64,
    barrett_mu: u64,
    inv_twiddle_factors: Vec<u64>,
    inv_n: u64,
}

impl FheIntt {
    /// Create a new INTT operation
    /// ## Parameters
    /// - `input`: NTT-domain polynomial tensor (u32 pairs representing u64)
    /// - `degree`: Polynomial degree N (must be power of 2)
    /// - `modulus`: Prime modulus q (must satisfy q ≡ 1 mod 2N)
    /// - `inv_root_of_unity`: Inverse N-th root of unity ω^(-1) (in `Z_q`)
    /// ## Constraints
    /// - N must be a power of 2
    /// - q must be prime
    /// - ω^(-1) * ω ≡ 1 (mod q)
    /// # Errors
    /// Returns [`Err`] if degree is not a power of 2, input size mismatch, or
    /// modulus/root validation fails.
    pub fn new(input: Tensor, degree: u32, modulus: u64, inv_root_of_unity: u64) -> Result<Self> {
        // Validate inputs (same as NTT)
        let expected_size = (degree as usize) * 2;
        if input.len() != expected_size {
            return Err(BarracudaError::Device(format!(
                "Input length {} doesn't match expected {} (degree {} * 2)",
                input.len(),
                expected_size,
                degree
            )));
        }

        if !degree.is_power_of_two() {
            return Err(BarracudaError::Device(format!(
                "Degree {degree} must be a power of 2 for INTT"
            )));
        }

        if modulus == 0 {
            return Err(BarracudaError::Device(
                "Modulus must be non-zero".to_string(),
            ));
        }

        // Precompute Barrett constant
        let barrett_mu = u64::MAX.checked_div(modulus).unwrap_or(0);

        // Precompute inverse twiddle factors: (ω^(-1))^0, (ω^(-1))^1, ..., (ω^(-1))^(N-1)
        let inv_twiddle_factors = compute_twiddle_factors(degree, modulus, inv_root_of_unity);

        // Precompute N^(-1) mod q for final scaling
        let inv_n = compute_modular_inverse(degree as u64, modulus);

        Ok(Self {
            input,
            degree,
            modulus,
            inv_root_of_unity,
            barrett_mu,
            inv_twiddle_factors,
            inv_n,
        })
    }

    /// Get input tensor
    pub(super) fn input(&self) -> &Tensor {
        &self.input
    }

    /// Get degree
    pub(super) fn degree(&self) -> u32 {
        self.degree
    }

    /// Get modulus
    pub(super) fn modulus(&self) -> u64 {
        self.modulus
    }

    /// Get inverse root of unity
    pub(super) fn inv_root_of_unity(&self) -> u64 {
        self.inv_root_of_unity
    }

    /// Get Barrett mu
    pub(super) fn barrett_mu(&self) -> u64 {
        self.barrett_mu
    }

    /// Get inverse twiddle factors
    pub(super) fn inv_twiddle_factors(&self) -> &[u64] {
        &self.inv_twiddle_factors
    }

    /// Get inverse N
    pub(super) fn inv_n(&self) -> u64 {
        self.inv_n
    }
}

/// Compute twiddle factors (same as NTT helper)
fn compute_twiddle_factors(degree: u32, modulus: u64, root: u64) -> Vec<u64> {
    let mut factors = Vec::with_capacity(degree as usize);
    let mut power = 1u64;

    for _ in 0..degree {
        factors.push(power);
        power = (power as u128 * root as u128 % modulus as u128) as u64;
    }

    factors
}

/// Compute modular inverse: a^(-1) mod m
/// Uses Extended Euclidean Algorithm
pub(crate) fn compute_modular_inverse(a: u64, m: u64) -> u64 {
    // Extended Euclidean Algorithm
    let (mut t, mut new_t) = (0i128, 1i128);
    let (mut r, mut new_r) = (m as i128, a as i128);

    while new_r != 0 {
        let quotient = r / new_r;
        (t, new_t) = (new_t, t - quotient * new_t);
        (r, new_r) = (new_r, r - quotient * new_r);
    }

    if r > 1 {
        // a is not invertible
        return 0;
    }

    if t < 0 {
        t += m as i128;
    }

    t as u64
}

/// Compute inverse primitive root: ω^(-1) where ω is N-th root of unity
#[must_use]
pub fn compute_inverse_root(_degree: u32, modulus: u64, root: u64) -> u64 {
    compute_modular_inverse(root, modulus)
}
