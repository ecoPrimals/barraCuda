// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for stats.fit_quadratic, stats.fit_exponential, stats.fit_logarithmic,
//! stats.rarefaction_curve, stats.gamma_fit, stats.gamma_cdf IPC methods.

use super::*;

// ── stats.fit_quadratic ──────────────────────────────────────────────────

#[test]
fn test_stats_fit_quadratic() {
    let x: Vec<f64> = (0..10).map(f64::from).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&xi| (2.0 * xi).mul_add(xi, 3.0f64.mul_add(xi, 1.0)))
        .collect();
    let resp = stats_fit_quadratic(&serde_json::json!({"x": x, "y": y}), serde_json::json!(1));
    let result = resp.result.expect("quadratic fit should succeed");
    assert_eq!(result["model"], "quadratic");
    assert!(result["r_squared"].as_f64().unwrap() > 0.999);
}

#[test]
fn test_stats_fit_quadratic_too_few_points() {
    let resp = stats_fit_quadratic(
        &serde_json::json!({"x": [1.0, 2.0], "y": [1.0, 4.0]}),
        serde_json::json!(2),
    );
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_stats_fit_quadratic() {
    let primal = test_primal();
    let x: Vec<f64> = (0..5).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
    let resp = dispatch(
        &primal,
        "stats.fit_quadratic",
        &serde_json::json!({"x": x, "y": y}),
        serde_json::json!(3),
    )
    .await;
    assert!(resp.result.is_some());
}

// ── stats.fit_exponential ────────────────────────────────────────────────

#[test]
fn test_stats_fit_exponential() {
    let x: Vec<f64> = (1..8).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * (0.5 * xi).exp()).collect();
    let resp = stats_fit_exponential(&serde_json::json!({"x": x, "y": y}), serde_json::json!(10));
    let result = resp.result.expect("exponential fit should succeed");
    assert_eq!(result["model"], "exponential");
    assert!(result["r_squared"].as_f64().unwrap() > 0.99);
}

#[test]
fn test_stats_fit_exponential_no_positive_y() {
    let resp = stats_fit_exponential(
        &serde_json::json!({"x": [1.0, 2.0], "y": [-1.0, -2.0]}),
        serde_json::json!(11),
    );
    assert!(resp.error.is_some());
}

// ── stats.fit_logarithmic ────────────────────────────────────────────────

#[test]
fn test_stats_fit_logarithmic() {
    let x: Vec<f64> = (1..10).map(f64::from).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 3.0f64.mul_add(xi.ln(), 5.0)).collect();
    let resp = stats_fit_logarithmic(&serde_json::json!({"x": x, "y": y}), serde_json::json!(20));
    let result = resp.result.expect("logarithmic fit should succeed");
    assert_eq!(result["model"], "logarithmic");
    assert!(result["r_squared"].as_f64().unwrap() > 0.999);
}

// ── stats.rarefaction_curve ──────────────────────────────────────────────

#[test]
fn test_stats_rarefaction_curve_monotone() {
    let counts = vec![10.0, 20.0, 30.0, 5.0, 15.0];
    let depths = vec![10.0, 20.0, 40.0, 80.0];
    let resp = stats_rarefaction_curve(
        &serde_json::json!({"counts": counts, "depths": depths}),
        serde_json::json!(30),
    );
    let result = resp.result.expect("rarefaction should succeed");
    let curve: Vec<f64> = result["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(curve.len(), 4);
    for w in curve.windows(2) {
        assert!(
            w[1] >= w[0],
            "rarefaction must be monotonically non-decreasing"
        );
    }
}

#[test]
fn test_stats_rarefaction_curve_missing_params() {
    let resp = stats_rarefaction_curve(
        &serde_json::json!({"counts": [1.0, 2.0]}),
        serde_json::json!(31),
    );
    assert!(resp.error.is_some());
}

// ── stats.gamma_fit ──────────────────────────────────────────────────────

#[test]
fn test_stats_gamma_fit() {
    let data = vec![2.1, 3.5, 1.8, 4.2, 2.9, 3.1, 2.5, 1.6, 3.8, 2.7];
    let resp = stats_gamma_fit(&serde_json::json!({"data": data}), serde_json::json!(40));
    let result = resp.result.expect("gamma fit should succeed");
    let alpha = result["alpha"].as_f64().unwrap();
    let beta = result["beta"].as_f64().unwrap();
    assert!(alpha > 0.0, "alpha must be positive");
    assert!(beta > 0.0, "beta must be positive");
    assert_eq!(result["n_positive"].as_u64().unwrap(), 10);
}

#[test]
fn test_stats_gamma_fit_too_few() {
    let resp = stats_gamma_fit(
        &serde_json::json!({"data": [1.0, 2.0]}),
        serde_json::json!(41),
    );
    assert!(resp.error.is_some());
}

// ── stats.gamma_cdf ──────────────────────────────────────────────────────

#[test]
fn test_stats_gamma_cdf_zero() {
    let resp = stats_gamma_cdf(
        &serde_json::json!({"x": 0.0, "alpha": 2.0, "beta": 1.0}),
        serde_json::json!(50),
    );
    let result = resp.result.expect("gamma_cdf at 0 should succeed");
    assert!((result["result"].as_f64().unwrap()).abs() < 1e-15);
}

#[test]
fn test_stats_gamma_cdf_large_x() {
    let resp = stats_gamma_cdf(
        &serde_json::json!({"x": 100.0, "alpha": 2.0, "beta": 1.0}),
        serde_json::json!(51),
    );
    let result = resp.result.expect("gamma_cdf should succeed");
    let p = result["result"].as_f64().unwrap();
    assert!(p > 0.99, "CDF at large x should be near 1, got {p}");
}

#[tokio::test]
async fn test_dispatch_stats_gamma_cdf() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.gamma_cdf",
        &serde_json::json!({"x": 5.0, "alpha": 2.0, "beta": 1.0}),
        serde_json::json!(52),
    )
    .await;
    assert!(resp.result.is_some());
}
