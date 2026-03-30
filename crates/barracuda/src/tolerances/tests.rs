// SPDX-License-Identifier: AGPL-3.0-or-later

use super::precision::*;
use super::registry::*;
use super::*;

#[test]
fn check_abs_tol() {
    assert!(check(1.0, 1.0 + 1e-15, &LINALG_TRANSPOSE));
    assert!(!check(1.0, 1.0 + 1e-10, &LINALG_TRANSPOSE));
}

#[test]
fn check_rel_tol() {
    assert!(check(100.0, 100.0 + 1e-8, &LINALG_MATMUL));
    assert!(!check(100.0, 100.0 + 1e-5, &LINALG_MATMUL));
}

#[test]
fn check_zero_expected() {
    assert!(check(1e-15, 0.0, &LINALG_TRANSPOSE));
}

#[test]
fn check_nan_rejects() {
    assert!(!check(f64::NAN, 1.0, &LINALG_MATMUL));
    assert!(!check(1.0, f64::NAN, &LINALG_MATMUL));
}

#[test]
fn check_infinity() {
    assert!(check(f64::INFINITY, f64::INFINITY, &LINALG_MATMUL));
}

#[test]
fn tiered_tolerances_ordered() {
    const { assert!(DETERMINISM.abs_tol <= MACHINE.abs_tol) };
    const { assert!(MACHINE.abs_tol <= ACCUMULATION.abs_tol) };
    const { assert!(ACCUMULATION.abs_tol <= TRANSCENDENTAL.abs_tol) };
    const { assert!(TRANSCENDENTAL.abs_tol <= ITERATIVE.abs_tol) };
    const { assert!(ITERATIVE.abs_tol <= STATISTICAL.abs_tol) };
    const { assert!(STATISTICAL.abs_tol <= STOCHASTIC.abs_tol) };
    const { assert!(STOCHASTIC.abs_tol <= EQUILIBRIUM.abs_tol) };
}

#[test]
fn eps_midpoint_safe() {
    assert_eq!(eps::midpoint(0.0, 10.0), 5.0);
    assert_eq!(eps::midpoint(-1.0, 1.0), 0.0);
    let large = f64::MAX * 0.5;
    let result = eps::midpoint(large, large);
    assert!(result.is_finite());
}

#[test]
fn all_tolerances_registry() {
    let all = all_tolerances();
    assert!(
        all.len() >= 51,
        "registry should have 51+ tolerances (36 domain + 15 precision), got {}",
        all.len()
    );
    for t in all {
        assert!(!t.name.is_empty());
        assert!(!t.justification.is_empty());
        assert!(t.abs_tol.is_finite());
        assert!(t.rel_tol.is_finite());
    }
}

#[test]
fn precision_tier_tolerance_ordering() {
    const { assert!(PRECISION_DF128.abs_tol < PRECISION_QF128.abs_tol) };
    const { assert!(PRECISION_QF128.abs_tol < PRECISION_F64_PRECISE.abs_tol) };
    const { assert!(PRECISION_F64_PRECISE.abs_tol < PRECISION_F64.abs_tol) };
    const { assert!(PRECISION_F64.abs_tol < PRECISION_DF64.abs_tol) };
    const { assert!(PRECISION_DF64.abs_tol < PRECISION_F32.abs_tol) };
    const { assert!(PRECISION_F32.abs_tol < PRECISION_F16.abs_tol) };
    const { assert!(PRECISION_F16.abs_tol < PRECISION_BF16.abs_tol) };
    const { assert!(PRECISION_BF16.abs_tol < PRECISION_FP8_E4M3.abs_tol) };
    const { assert!(PRECISION_FP8_E4M3.abs_tol < PRECISION_FP8_E5M2.abs_tol) };
}

#[test]
fn precision_tier_lookup_all() {
    use crate::device::precision_tier::PrecisionTier;
    let expected: &[(PrecisionTier, &str)] = &[
        (PrecisionTier::Binary, "precision::Binary"),
        (PrecisionTier::Int2, "precision::INT2"),
        (PrecisionTier::Quantized4, "precision::Q4"),
        (PrecisionTier::Quantized8, "precision::Q8"),
        (PrecisionTier::Fp8E5M2, "precision::FP8_E5M2"),
        (PrecisionTier::Fp8E4M3, "precision::FP8_E4M3"),
        (PrecisionTier::Bf16, "precision::BF16"),
        (PrecisionTier::F16, "precision::F16"),
        (PrecisionTier::Tf32, "precision::TF32"),
        (PrecisionTier::F32, "precision::F32"),
        (PrecisionTier::DF64, "precision::DF64"),
        (PrecisionTier::F64, "precision::F64"),
        (PrecisionTier::F64Precise, "precision::F64Precise"),
        (PrecisionTier::QF128, "precision::QF128"),
        (PrecisionTier::DF128, "precision::DF128"),
    ];
    for &(tier, name) in expected {
        let tol = for_precision_tier(tier);
        assert_eq!(tol.name, name, "for_precision_tier({tier:?}) mismatch");
        assert!(tol.abs_tol.is_finite());
        assert!(tol.rel_tol.is_finite());
    }
}

#[test]
fn by_name_lookup() {
    assert_eq!(by_name("pharma_foce").unwrap().name, "pharma_foce");
    assert_eq!(by_name("signal_fft").unwrap().name, "signal_fft");
    assert_eq!(by_name("precision::F32").unwrap().name, "precision::F32");
    assert_eq!(
        by_name("precision::DF128").unwrap().name,
        "precision::DF128"
    );
    assert!(by_name("nonexistent").is_none());
}

#[test]
fn tier_lookup_all() {
    let tiers = [
        ("determinism", "tol::DETERMINISM"),
        ("machine", "tol::MACHINE"),
        ("accumulation", "tol::ACCUMULATION"),
        ("transcendental", "tol::TRANSCENDENTAL"),
        ("iterative", "tol::ITERATIVE"),
        ("statistical", "tol::STATISTICAL"),
        ("stochastic", "tol::STOCHASTIC"),
        ("equilibrium", "tol::EQUILIBRIUM"),
    ];
    for (category, expected_name) in tiers {
        let t = tier(category).unwrap_or_else(|| panic!("tier({category}) returned None"));
        assert_eq!(t.name, expected_name);
    }
    assert!(tier("nonexistent").is_none());
}

#[test]
fn eps_guards_positive() {
    const { assert!(eps::SAFE_DIV > 0.0) };
    const { assert!(eps::LOG_FLOOR > 0.0) };
    const { assert!(eps::PROB_FLOOR > 0.0) };
}

#[test]
fn tolerances_have_finite_values() {
    for t in all_tolerances() {
        assert!(t.abs_tol.is_finite(), "{} abs_tol must be finite", t.name);
        assert!(t.rel_tol.is_finite(), "{} rel_tol must be finite", t.name);
        assert!(!t.name.is_empty(), "tolerance name must not be empty");
        assert!(
            !t.justification.is_empty(),
            "justification must not be empty"
        );
    }
}
