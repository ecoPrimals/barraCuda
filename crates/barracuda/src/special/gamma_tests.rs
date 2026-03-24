// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use std::f64::consts::PI;

#[test]
fn test_ln_gamma_integers() {
    assert!((ln_gamma(1.0).unwrap() - 0.0).abs() < 1e-10);
    assert!((ln_gamma(2.0).unwrap() - 0.0).abs() < 1e-10);
    assert!((ln_gamma(3.0).unwrap() - 2.0_f64.ln()).abs() < 1e-10);
    assert!((ln_gamma(4.0).unwrap() - 6.0_f64.ln()).abs() < 1e-10);
    assert!((ln_gamma(5.0).unwrap() - 24.0_f64.ln()).abs() < 1e-10);
}

#[test]
fn test_gamma_integers() {
    assert!((gamma(1.0).unwrap() - 1.0).abs() < 1e-10);
    assert!((gamma(2.0).unwrap() - 1.0).abs() < 1e-10);
    assert!((gamma(3.0).unwrap() - 2.0).abs() < 1e-10);
    assert!((gamma(4.0).unwrap() - 6.0).abs() < 1e-10);
    assert!((gamma(5.0).unwrap() - 24.0).abs() < 1e-10);
}

#[test]
fn test_gamma_half_integer() {
    assert!((gamma(0.5).unwrap() - PI.sqrt()).abs() < 1e-10);
    assert!((gamma(1.5).unwrap() - PI.sqrt() / 2.0).abs() < 1e-10);
}

#[test]
fn test_gamma_invalid_input() {
    assert!(gamma(0.0).is_err());
    assert!(gamma(-1.0).is_err());
    assert!(gamma(-2.0).is_err());
}

#[test]
fn test_regularized_gamma_p_exponential() {
    let p = regularized_gamma_p(1.0, 1.0).unwrap();
    assert!((p - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);

    let p = regularized_gamma_p(1.0, 2.0).unwrap();
    assert!((p - (1.0 - (-2.0_f64).exp())).abs() < 1e-10);
}

#[test]
fn test_regularized_gamma_p_bounds() {
    assert!((regularized_gamma_p(2.0, 0.0).unwrap() - 0.0).abs() < 1e-10);

    let p_large = regularized_gamma_p(2.0, 50.0).unwrap();
    assert!(p_large > 0.9999999);
}

#[test]
fn test_regularized_gamma_q_complement() {
    let a = 2.5;
    let x = 3.0;
    let p = regularized_gamma_p(a, x).unwrap();
    let q = regularized_gamma_q(a, x).unwrap();

    assert!((p + q - 1.0).abs() < 1e-10);
}

#[test]
fn test_incomplete_gamma_relation() {
    let a = 2.0;
    let x = 1.5;

    let (lower, complete) = lower_incomplete_gamma(a, x).unwrap();
    let upper = upper_incomplete_gamma(a, x).unwrap();

    assert!((lower + upper - complete).abs() < 1e-10);
}

#[test]
fn test_gamma_series_small_x() {
    let p = regularized_gamma_p(3.0, 1.0).unwrap();
    assert!(p > 0.0 && p < 1.0);
}

#[test]
fn test_gamma_cf_large_x() {
    let p = regularized_gamma_p(2.0, 5.0).unwrap();
    assert!(p > 0.9 && p < 1.0);
}

#[test]
fn test_digamma_euler_mascheroni() {
    const EULER_MASCHERONI: f64 = 0.5772156649015329;
    let psi_1 = digamma(1.0).unwrap();
    assert!(
        (psi_1 - (-EULER_MASCHERONI)).abs() < 1e-9,
        "ψ(1) = {}, expected {}",
        psi_1,
        -EULER_MASCHERONI
    );
}

#[test]
fn test_digamma_recurrence() {
    let psi_1 = digamma(1.0).unwrap();
    let psi_2 = digamma(2.0).unwrap();
    assert!(
        (psi_2 - (psi_1 + 1.0)).abs() < 1e-10,
        "ψ(2) = {}, expected ψ(1) + 1 = {}",
        psi_2,
        psi_1 + 1.0
    );

    let psi_3 = digamma(3.0).unwrap();
    assert!(
        (psi_3 - (psi_2 + 0.5)).abs() < 1e-10,
        "ψ(3) = {}, expected ψ(2) + 1/2 = {}",
        psi_3,
        psi_2 + 0.5
    );
}

#[test]
fn test_digamma_half() {
    const EULER_MASCHERONI: f64 = 0.5772156649015329;
    let expected = 2.0f64.mul_add(-2.0_f64.ln(), -EULER_MASCHERONI);
    let psi_half = digamma(0.5).unwrap();
    assert!(
        (psi_half - expected).abs() < 1e-9,
        "ψ(1/2) = {psi_half}, expected {expected}"
    );
}

#[test]
fn test_digamma_invalid_input() {
    assert!(digamma(0.0).is_err());
    assert!(digamma(-1.0).is_err());
}

#[test]
fn test_beta_simple_values() {
    let b_11 = beta(1.0, 1.0).unwrap();
    assert!((b_11 - 1.0).abs() < 1e-10);

    let b_31 = beta(3.0, 1.0).unwrap();
    assert!((b_31 - 1.0 / 3.0).abs() < 1e-10);

    let b_14 = beta(1.0, 4.0).unwrap();
    assert!((b_14 - 0.25).abs() < 1e-10);
}

#[test]
fn test_beta_symmetry() {
    let b_ab = beta(2.5, 3.7).unwrap();
    let b_ba = beta(3.7, 2.5).unwrap();
    assert!((b_ab - b_ba).abs() < 1e-12);
}

#[test]
fn test_beta_half_half() {
    let b_half = beta(0.5, 0.5).unwrap();
    assert!(
        (b_half - PI).abs() < 1e-10,
        "B(1/2, 1/2) = {b_half}, expected π = {PI}"
    );
}

#[test]
fn test_beta_integers() {
    let b_23 = beta(2.0, 3.0).unwrap();
    assert!((b_23 - 1.0 / 12.0).abs() < 1e-10);

    let b_34 = beta(3.0, 4.0).unwrap();
    assert!((b_34 - 1.0 / 60.0).abs() < 1e-10);
}

#[test]
fn test_beta_invalid_input() {
    assert!(beta(0.0, 1.0).is_err());
    assert!(beta(1.0, 0.0).is_err());
    assert!(beta(-1.0, 2.0).is_err());
}

#[test]
fn test_ln_beta() {
    let lb_11 = ln_beta(1.0, 1.0).unwrap();
    assert!(lb_11.abs() < 1e-10);

    let a = 2.5;
    let b = 3.7;
    let lb = ln_beta(a, b).unwrap();
    let b_val = beta(a, b).unwrap();
    assert!((lb - b_val.ln()).abs() < 1e-12);
}
