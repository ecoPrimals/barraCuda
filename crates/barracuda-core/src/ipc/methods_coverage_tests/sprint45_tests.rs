// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for Sprint 45 JSON-RPC surface expansion:
//! stats.chi_squared, stats.anova_oneway, linalg.svd, linalg.qr,
//! activation.softmax, activation.gelu, spectral.stft, and aliases.

use crate::ipc::jsonrpc;

use super::super::math::{
    activation_gelu, activation_softmax, linalg_eigenvalues, linalg_qr, linalg_svd,
    stats_anova_oneway, stats_chi_squared, stats_correlation,
};
use super::super::spectral::spectral_stft;

// ── stats.eigh alias ───────────────────────────────────────────────

#[test]
fn stats_eigh_is_eigenvalues() {
    let params = serde_json::json!({"matrix": [[2.0, 1.0], [1.0, 3.0]]});
    let resp = linalg_eigenvalues(&params, serde_json::json!(200));
    assert!(resp.error.is_none());
    let eigenvalues = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(eigenvalues.len(), 2);
}

// ── stats.pearson alias ────────────────────────────────────────────

#[test]
fn stats_pearson_is_correlation() {
    let params = serde_json::json!({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]});
    let resp = stats_correlation(&params, serde_json::json!(201));
    assert!(resp.error.is_none());
    let r = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((r - 1.0).abs() < 1e-10);
}

// ── stats.chi_squared ──────────────────────────────────────────────

#[test]
fn stats_chi_squared_missing_observed() {
    let resp = stats_chi_squared(
        &serde_json::json!({"expected": [10.0, 10.0]}),
        serde_json::json!(210),
    );
    let err = resp.error.expect("missing observed");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_chi_squared_missing_expected() {
    let resp = stats_chi_squared(
        &serde_json::json!({"observed": [10.0, 10.0]}),
        serde_json::json!(211),
    );
    let err = resp.error.expect("missing expected");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_chi_squared_happy_path() {
    let resp = stats_chi_squared(
        &serde_json::json!({"observed": [10.0, 20.0, 30.0], "expected": [20.0, 20.0, 20.0]}),
        serde_json::json!(212),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let chi2 = r["chi_squared"].as_f64().unwrap();
    assert!(chi2 > 0.0);
    assert!(r["p_value"].as_f64().unwrap().is_finite());
    assert_eq!(r["df"].as_u64().unwrap(), 2);
}

// ── stats.anova_oneway ─────────────────────────────────────────────

#[test]
fn stats_anova_missing_groups() {
    let resp = stats_anova_oneway(&serde_json::json!({}), serde_json::json!(220));
    let err = resp.error.expect("missing groups");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_anova_single_group() {
    let resp = stats_anova_oneway(
        &serde_json::json!({"groups": [[1.0, 2.0]]}),
        serde_json::json!(221),
    );
    let err = resp.error.expect("need at least 2 groups");
    assert!(err.message.contains("2 groups"));
}

#[test]
fn stats_anova_empty_group() {
    let resp = stats_anova_oneway(
        &serde_json::json!({"groups": [[1.0, 2.0], []]}),
        serde_json::json!(222),
    );
    let err = resp.error.expect("empty group");
    assert!(
        err.message.contains("non-empty") || err.message.contains("must exceed"),
        "Expected validation error, got: {}",
        err.message
    );
}

#[test]
fn stats_anova_happy_path() {
    let resp = stats_anova_oneway(
        &serde_json::json!({"groups": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]}),
        serde_json::json!(223),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let f = r["f_statistic"].as_f64().unwrap();
    assert!(f > 0.0);
    let p = r["p_value"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&p));
    assert_eq!(r["df_between"].as_u64().unwrap(), 2);
    assert_eq!(r["df_within"].as_u64().unwrap(), 6);
}

#[test]
fn stats_anova_identical_groups() {
    let resp = stats_anova_oneway(
        &serde_json::json!({"groups": [[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]]}),
        serde_json::json!(224),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let p = r["p_value"].as_f64().unwrap();
    assert!(p > 0.99, "Identical groups should have p ≈ 1.0, got {p}");
}

// ── linalg.svd ─────────────────────────────────────────────────────

#[test]
fn linalg_svd_missing_matrix() {
    let resp = linalg_svd(&serde_json::json!({}), serde_json::json!(230));
    let err = resp.error.expect("missing matrix");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn linalg_svd_empty_matrix() {
    let resp = linalg_svd(&serde_json::json!({"matrix": []}), serde_json::json!(231));
    let err = resp.error.expect("empty matrix");
    assert!(err.message.contains("non-empty"));
}

#[test]
fn linalg_svd_happy_path() {
    let resp = linalg_svd(
        &serde_json::json!({"matrix": [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]]}),
        serde_json::json!(232),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let s = r["s"].as_array().unwrap();
    assert_eq!(s.len(), 2);
    let s0 = s[0].as_f64().unwrap();
    let s1 = s[1].as_f64().unwrap();
    assert!(s0 >= s1, "Singular values should be descending");
    assert!((s0 - 2.0).abs() < 1e-8);
    assert!((s1 - 1.0).abs() < 1e-8);
}

// ── linalg.qr ──────────────────────────────────────────────────────

#[test]
fn linalg_qr_missing_matrix() {
    let resp = linalg_qr(&serde_json::json!({}), serde_json::json!(240));
    let err = resp.error.expect("missing matrix");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn linalg_qr_happy_path() {
    let resp = linalg_qr(
        &serde_json::json!({"matrix": [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]}),
        serde_json::json!(241),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert_eq!(r["m"].as_u64().unwrap(), 3);
    assert_eq!(r["n"].as_u64().unwrap(), 2);
    let q = r["q"].as_array().unwrap();
    assert!(!q.is_empty());
}

// ── activation.softmax ─────────────────────────────────────────────

#[test]
fn activation_softmax_missing_data() {
    let resp = activation_softmax(&serde_json::json!({}), serde_json::json!(250));
    let err = resp.error.expect("missing data");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn activation_softmax_empty_data() {
    let resp = activation_softmax(&serde_json::json!({"data": []}), serde_json::json!(251));
    let err = resp.error.expect("empty data");
    assert!(err.message.contains("non-empty"));
}

#[test]
fn activation_softmax_happy_path() {
    let resp = activation_softmax(
        &serde_json::json!({"data": [1.0, 2.0, 3.0]}),
        serde_json::json!(252),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 3);
    let sum: f64 = arr.iter().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 1.0).abs() < 1e-10, "softmax should sum to 1");
    assert!(arr[2].as_f64().unwrap() > arr[1].as_f64().unwrap());
}

// ── activation.gelu ────────────────────────────────────────────────

#[test]
fn activation_gelu_missing_data() {
    let resp = activation_gelu(&serde_json::json!({}), serde_json::json!(260));
    let err = resp.error.expect("missing data");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn activation_gelu_happy_path() {
    let resp = activation_gelu(
        &serde_json::json!({"data": [0.0, 1.0, -1.0]}),
        serde_json::json!(261),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 3);
    assert!((arr[0].as_f64().unwrap()).abs() < 1e-10, "gelu(0) = 0");
    assert!(arr[1].as_f64().unwrap() > 0.8, "gelu(1) ≈ 0.841");
}

// ── spectral.stft ──────────────────────────────────────────────────

#[test]
fn spectral_stft_missing_data() {
    let resp = spectral_stft(&serde_json::json!({}), serde_json::json!(270));
    let err = resp.error.expect("missing data");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn spectral_stft_too_short() {
    let resp = spectral_stft(
        &serde_json::json!({"data": [1.0, 2.0], "n_fft": 4}),
        serde_json::json!(271),
    );
    let err = resp.error.expect("data too short");
    assert!(err.message.contains("n_fft"));
}

#[test]
fn spectral_stft_non_power_of_two() {
    let data = vec![0.0_f64; 300];
    let resp = spectral_stft(
        &serde_json::json!({"data": data, "n_fft": 300}),
        serde_json::json!(272),
    );
    let err = resp.error.expect("non-power-of-2 n_fft");
    assert!(err.message.contains("power of 2"));
}

#[test]
fn spectral_stft_happy_path() {
    let signal: Vec<f64> = (0..512).map(|i| (f64::from(i) * 0.1).sin()).collect();
    let resp = spectral_stft(
        &serde_json::json!({"data": signal, "n_fft": 64, "hop_length": 32}),
        serde_json::json!(273),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert_eq!(r["n_fft"].as_u64().unwrap(), 64);
    assert_eq!(r["hop_length"].as_u64().unwrap(), 32);
    let mag = r["magnitude"].as_array().unwrap();
    assert!(!mag.is_empty());
    let freq_bins = r["freq_bins"].as_u64().unwrap();
    assert_eq!(freq_bins, 33); // 64/2 + 1
}
