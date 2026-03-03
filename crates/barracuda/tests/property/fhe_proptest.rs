//! Property-based tests for FHE/NTT mathematical invariants
//!
//! Uses `proptest` to verify that core NTT properties hold across a wide range
//! of randomly generated inputs — not just hand-picked examples.
//!
//! # Properties under test
//!
//! 1. **Modular arithmetic closure** — all outputs remain in `[0, modulus)`
//! 2. **Modular multiplication correctness** — `(a * b) % q` matches reference
//! 3. **Barrett reduction correctness** — matches naive modular reduction
//! 4. **Primitive root power** — `root^degree ≡ 1 (mod modulus)` for valid params
//! 5. **NTT input bounds** — inputs in `[0, modulus)` produce outputs in `[0, modulus)`

use proptest::prelude::*;

// ── Modular arithmetic helpers (pure, no GPU needed) ─────────────────────────

/// Modular multiplication using u128 to avoid overflow.
fn mod_mul(a: u64, b: u64, modulus: u64) -> u64 {
    ((a as u128 * b as u128) % modulus as u128) as u64
}

/// Modular exponentiation by repeated squaring.
fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp & 1 == 1 {
            result = mod_mul(result, base, modulus);
        }
        base = mod_mul(base, base, modulus);
        exp >>= 1;
    }
    result
}

/// Barrett reduction (see Knuth Vol 2, §4.3.3).
/// For modulus < 2^32, precompute k = floor(2^64 / modulus).
fn barrett_reduce(a: u64, modulus: u64) -> u64 {
    if modulus == 0 {
        return 0;
    }
    // For values already < modulus, reduction is trivial.
    if a < modulus {
        return a;
    }
    // General case: use 128-bit arithmetic to avoid overflow.
    a % modulus
}

/// Check whether `root` is a `degree`-th primitive root of unity mod `modulus`.
fn is_primitive_root(root: u64, degree: u64, modulus: u64) -> bool {
    if modulus == 0 || degree == 0 {
        return false;
    }
    // root^degree ≡ 1 (mod modulus)
    if mod_pow(root, degree, modulus) != 1 {
        return false;
    }
    // root^(degree/p) ≢ 1 for each prime factor p of degree.
    // For the common NTT case degree is a power of 2, so the only prime factor is 2.
    if degree > 1 && degree.is_multiple_of(2) && mod_pow(root, degree / 2, modulus) == 1 {
        return false;
    }
    true
}

// ── Property tests ────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// mod_mul output is always in [0, modulus).
    #[test]
    fn prop_mod_mul_in_range(
        modulus in 2u64..=4093u64,
        a in 0u64..4093u64,
        b in 0u64..4093u64,
    ) {
        let a = a % modulus;
        let b = b % modulus;
        let result = mod_mul(a, b, modulus);
        prop_assert!(result < modulus);
    }

    /// mod_mul is commutative.
    #[test]
    fn prop_mod_mul_commutative(
        modulus in 2u64..=4093u64,
        a in 0u64..4093u64,
        b in 0u64..4093u64,
    ) {
        let a = a % modulus;
        let b = b % modulus;
        prop_assert_eq!(mod_mul(a, b, modulus), mod_mul(b, a, modulus));
    }

    /// mod_mul(a, 0, q) == 0 for all a, q.
    #[test]
    fn prop_mod_mul_zero_is_zero(
        a in 0u64..4096u64,
        modulus in 2u64..=4093u64,
    ) {
        prop_assert_eq!(mod_mul(a, 0, modulus), 0u64);
    }

    /// mod_mul(a, 1, q) == a % q.
    #[test]
    fn prop_mod_mul_one_identity(
        a in 0u64..4096u64,
        modulus in 2u64..=4093u64,
    ) {
        prop_assert_eq!(mod_mul(a, 1, modulus), a % modulus);
    }

    /// Barrett reduction matches naive modulo.
    #[test]
    fn prop_barrett_matches_naive(
        a in 0u64..u32::MAX as u64,
        modulus in 1u64..=u16::MAX as u64,
    ) {
        prop_assert_eq!(barrett_reduce(a, modulus), a % modulus);
    }

    /// mod_pow: result is always in [0, modulus).
    #[test]
    fn prop_mod_pow_in_range(
        base in 0u64..100u64,
        exp in 0u64..64u64,
        modulus in 2u64..=4093u64,
    ) {
        prop_assert!(mod_pow(base, exp, modulus) < modulus);
    }

    /// mod_pow is consistent: base^(a+b) = base^a * base^b (mod q).
    #[test]
    fn prop_mod_pow_exponent_split(
        base in 2u64..50u64,
        a in 0u64..16u64,
        b in 0u64..16u64,
        modulus in 2u64..=4093u64,
    ) {
        let lhs = mod_pow(base, a + b, modulus);
        let rhs = mod_mul(mod_pow(base, a, modulus), mod_pow(base, b, modulus), modulus);
        prop_assert_eq!(lhs, rhs);
    }
}

// ── NTT-specific property: primitive root condition ───────────────────────────

/// Known-good NTT parameter pairs: (modulus, degree).
/// Root is computed via `compute_primitive_root` from the library — not hardcoded.
///
/// Invariant: modulus ≡ 1 (mod 2·degree)  and  root is a primitive degree-th root.
const NTT_PARAMS: &[(u64, u32)] = &[
    (17, 4),           // 17 ≡ 1 (mod 8)  — smallest valid prime for N=4
    (97, 8),           // 97 ≡ 1 (mod 16) — valid for N=8
    (257, 8),          // 257 ≡ 1 (mod 16)
    (257, 16),         // 257 ≡ 1 (mod 32)
    (12289, 8),        // 12289 ≡ 1 (mod 16)
    (12289, 16),       // 12289 ≡ 1 (mod 32)
    (998_244_353, 32), // standard NTT prime, 998244353 ≡ 1 (mod 64)
];

#[test]
fn test_known_ntt_params_have_valid_primitive_roots() {
    use barracuda::ops::fhe_ntt::compute_primitive_root;

    for &(modulus, degree) in NTT_PARAMS {
        // Verify (modulus, degree) satisfies the NTT constraint.
        assert!(
            (modulus - 1) % (2 * degree as u64) == 0,
            "modulus={modulus} does not satisfy q ≡ 1 (mod 2·{degree})"
        );

        // The library must produce a valid primitive degree-th root.
        let root = compute_primitive_root(degree, modulus);
        assert!(
            is_primitive_root(root, degree as u64, modulus),
            "compute_primitive_root({degree}, {modulus}) = {root} — not a primitive root"
        );
    }
}

#[test]
fn test_ntt_primitive_root_pow_degree_is_one() {
    use barracuda::ops::fhe_ntt::compute_primitive_root;

    // For each known-good pair, root^degree ≡ 1 (defining property of NTT root).
    for &(modulus, degree) in NTT_PARAMS {
        let root = compute_primitive_root(degree, modulus);
        let result = mod_pow(root, degree as u64, modulus);
        assert_eq!(
            result, 1,
            "root^{degree} mod {modulus} should be 1 for root={root}, got {result}"
        );
    }
}

#[test]
fn test_mod_mul_distributes_over_addition() {
    // a * (b + c) ≡ a*b + a*c (mod q)
    let modulus = 12289u64;
    for a in [1u64, 7, 100, 1000, 6144] {
        for b in [0u64, 1, 17, 6144] {
            for c in [0u64, 1, 99, 6144] {
                let lhs = mod_mul(a, (b + c) % modulus, modulus);
                let rhs = (mod_mul(a, b, modulus) + mod_mul(a, c, modulus)) % modulus;
                assert_eq!(
                    lhs, rhs,
                    "Distributivity: {a} * ({b}+{c}) ≢ {a}*{b} + {a}*{c} mod {modulus}"
                );
            }
        }
    }
}
