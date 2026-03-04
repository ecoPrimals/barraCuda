// SPDX-License-Identifier: AGPL-3.0-or-later
//! Test data generators and helpers for FHE shader unit tests.

use barracuda::ops::fhe_ntt::compute_primitive_root;
use barracuda::tensor::Tensor;

/// Generate random polynomial with coefficients in [0, modulus)
pub fn random_polynomial(degree: usize, modulus: u64) -> Vec<u64> {
    use std::collections::hash_map::RandomState;
    use std::hash::BuildHasher;

    let hasher_builder = RandomState::new();
    (0..degree)
        .map(|i| hasher_builder.hash_one(i) % modulus)
        .collect()
}

/// Known primitive roots for testing
/// Format: (degree, modulus, root_of_unity)
/// All satisfy: modulus ≡ 1 (mod 2*degree) and root^degree ≡ 1
pub const KNOWN_ROOTS: &[(u32, u64, u64)] = &[
    (4, 17, 4),       // 4^4 ≡ 1 mod 17, 17 ≡ 1 mod 8
    (4, 97, 22),      // 22^4 ≡ 1 mod 97
    (4, 257, 241),    // 241^4 ≡ 1 mod 257
    (4, 12289, 1479), // 1479^4 ≡ 1 mod 12289
    (8, 97, 64),      // 64^8 ≡ 1 mod 97 (was 10 - incorrect)
    (16, 97, 8),      // 8^16 ≡ 1 mod 97 (was 92 - incorrect)
];

/// Compute modular inverse: a^(-1) mod m
#[expect(dead_code, reason = "test utility for FHE INTT validation")]
pub fn mod_inverse(a: u64, m: u64) -> u64 {
    // Extended Euclidean algorithm
    let (mut old_r, mut r) = (a as i128, m as i128);
    let (mut old_s, mut s) = (1i128, 0i128);

    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
    }

    if old_s < 0 {
        (old_s + m as i128) as u64
    } else {
        old_s as u64
    }
}

/// Find a primitive N-th root of unity for given degree and modulus.
/// NTT requires: modulus ≡ 1 (mod 2*degree).
/// Uses barracuda's compute_primitive_root when the modulus satisfies this constraint.
pub fn find_root_of_unity(degree: u32, modulus: u64) -> Option<u64> {
    // Check known roots first
    for &(d, m, root) in KNOWN_ROOTS {
        if d == degree && m == modulus {
            return Some(root);
        }
    }

    // NTT requires q ≡ 1 (mod 2*degree). If satisfied, use compute_primitive_root.
    let two_n = 2u64 * degree as u64;
    if (modulus - 1).is_multiple_of(two_n) {
        let root = compute_primitive_root(degree, modulus);
        // Verify it's a valid primitive N-th root (omega^N = 1, omega^(N/2) != 1)
        let mut power = 1u64;
        for _ in 0..degree {
            power = (power as u128 * root as u128 % modulus as u128) as u64;
        }
        if power == 1 && root != 1 {
            return Some(root);
        }
    }

    // Fallback: try small candidates
    for candidate in 2..modulus.min(100) {
        let mut power = 1u64;
        for _ in 0..degree {
            power = (power as u128 * candidate as u128 % modulus as u128) as u64;
        }
        if power == 1 {
            let mut is_primitive = true;
            for k in 1..degree {
                let mut p = 1u64;
                for _ in 0..k {
                    p = (p as u128 * candidate as u128 % modulus as u128) as u64;
                }
                if p == 1 {
                    is_primitive = false;
                    break;
                }
            }
            if is_primitive {
                return Some(candidate);
            }
        }
    }

    None
}

/// Modulus/degree pairs that satisfy NTT constraint q ≡ 1 (mod 2*degree).
/// Use these for tests to avoid invalid combinations.
pub fn modulus_supports_degree(modulus: u64, degree: u32) -> bool {
    let two_n = 2u64 * degree as u64;
    (modulus - 1).is_multiple_of(two_n)
}

/// Helper to read tensor back as u64 polynomial
pub async fn read_poly_from_tensor(tensor: &Tensor) -> Vec<u64> {
    let u32_data = tensor
        .to_vec_u32()
        .expect("tensor conversion to u32 vec failed");
    u32_pairs_to_poly(&u32_data)
}

/// Convert u32 pairs from GPU back to u64 polynomial
pub fn u32_pairs_to_poly(pairs: &[u32]) -> Vec<u64> {
    pairs
        .chunks(2)
        .map(|chunk| {
            let low = chunk[0] as u64;
            let high = chunk[1] as u64;
            low | (high << 32)
        })
        .collect()
}

/// FHE-friendly primes for testing
#[expect(
    dead_code,
    reason = "shared test constants for parameterized FHE tests"
)]
pub const TEST_PRIMES: &[u64] = &[
    17,    // Tiny (for fast tests)
    97,    // Small
    12289, // Standard FHE prime
    65537, // Fermat prime
];

/// Common test degrees (powers of 2)
#[expect(
    dead_code,
    reason = "shared test constants for parameterized FHE tests"
)]
pub const TEST_DEGREES: &[usize] = &[4, 8, 16, 32, 64, 128, 256, 512, 1024];
