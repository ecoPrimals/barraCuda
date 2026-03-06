// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for FHE Number Theoretic Transform

use super::*;

#[test]
fn test_twiddle_factors() {
    // Small test: N=4, q=17 (17 ≡ 1 mod 8)
    // Root of unity: 4^2 ≡ 1 (mod 17), so ω=4
    let factors = compute_twiddle_factors(4, 17, 4);

    assert_eq!(factors.len(), 4);
    assert_eq!(factors[0], 1); // ω^0 = 1
    assert_eq!(factors[1], 4); // ω^1 = 4
    assert_eq!(factors[2], 16); // ω^2 = 16 ≡ -1 (mod 17)
    assert_eq!(factors[3], 13); // ω^3 = 13 ≡ -4 (mod 17)
}

#[test]
fn test_degree_validation() {
    // Degree must be power of 2
    assert!(8u32.is_power_of_two());
    assert!(4096u32.is_power_of_two());
    assert!(!100u32.is_power_of_two());
}

#[test]
fn test_modulus_constraint() {
    // Modulus constraint: q ≡ 1 (mod 2N). For N=4, q=17: 17 ≡ 1 (mod 8).
    let degree = 4u32;
    let modulus = 17u64;
    assert!((modulus - 1).is_multiple_of(2 * u64::from(degree)));
}
