// SPDX-License-Identifier: AGPL-3.0-only
//! Tests for FHE Inverse Number Theoretic Transform

use super::*;

#[test]
fn test_modular_inverse() {
    // Test: 3^(-1) mod 7 = 5 (because 3 * 5 = 15 ≡ 1 mod 7)
    assert_eq!(compute_modular_inverse(3, 7), 5);

    // Test: 4^(-1) mod 17 = 13 (because 4 * 13 = 52 ≡ 1 mod 17)
    assert_eq!(compute_modular_inverse(4, 17), 13);
}

#[test]
fn test_inverse_root() {
    // For N=4, q=17, ω=4: ω^(-1) = 4^(-1) mod 17 = 13
    assert_eq!(compute_inverse_root(4, 17, 4), 13);
}
