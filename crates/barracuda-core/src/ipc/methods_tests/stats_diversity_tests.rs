// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for stats.simpson, stats.bray_curtis, stats.hill IPC methods.

use super::*;

// ── stats.simpson ────────────────────────────────────────────────────────

#[test]
fn test_stats_simpson_uniform() {
    let resp = stats_simpson(
        &serde_json::json!({"counts": [25.0, 25.0, 25.0, 25.0]}),
        serde_json::json!(1),
    );
    let result = resp.result.expect("simpson should succeed");
    let d = result["result"].as_f64().unwrap();
    assert!(
        (d - 0.75).abs() < 1e-12,
        "4 equal species: D = 0.75, got {d}"
    );
}

#[test]
fn test_stats_simpson_singleton() {
    let resp = stats_simpson(
        &serde_json::json!({"counts": [100.0, 0.0, 0.0]}),
        serde_json::json!(2),
    );
    let result = resp.result.expect("simpson should succeed");
    let d = result["result"].as_f64().unwrap();
    assert!(d.abs() < 1e-12, "single species: D = 0, got {d}");
}

#[test]
fn test_stats_simpson_missing_params() {
    let resp = stats_simpson(&serde_json::json!({}), serde_json::json!(3));
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_dispatch_stats_simpson() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.simpson",
        &serde_json::json!({"counts": [10.0, 10.0, 10.0]}),
        serde_json::json!(4),
    )
    .await;
    let result = resp.result.expect("dispatch should succeed");
    assert_eq!(result["index"], "simpson");
}

// ── stats.bray_curtis ────────────────────────────────────────────────────

#[test]
fn test_stats_bray_curtis_identical() {
    let resp = stats_bray_curtis(
        &serde_json::json!({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}),
        serde_json::json!(10),
    );
    let result = resp.result.expect("bray_curtis should succeed");
    let d = result["result"].as_f64().unwrap();
    assert!(d.abs() < 1e-15, "identical samples: BC = 0, got {d}");
}

#[test]
fn test_stats_bray_curtis_disjoint() {
    let resp = stats_bray_curtis(
        &serde_json::json!({"a": [5.0, 0.0, 0.0], "b": [0.0, 0.0, 5.0]}),
        serde_json::json!(11),
    );
    let result = resp.result.expect("bray_curtis should succeed");
    let d = result["result"].as_f64().unwrap();
    assert!((d - 1.0).abs() < 1e-15, "disjoint samples: BC = 1, got {d}");
}

#[test]
fn test_stats_bray_curtis_length_mismatch() {
    let resp = stats_bray_curtis(
        &serde_json::json!({"a": [1.0, 2.0], "b": [1.0, 2.0, 3.0]}),
        serde_json::json!(12),
    );
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[test]
fn test_stats_bray_curtis_missing_params() {
    let resp = stats_bray_curtis(&serde_json::json!({"a": [1.0]}), serde_json::json!(13));
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_dispatch_stats_bray_curtis() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.bray_curtis",
        &serde_json::json!({"a": [3.0, 1.0], "b": [1.0, 3.0]}),
        serde_json::json!(14),
    )
    .await;
    let result = resp.result.expect("dispatch should succeed");
    assert_eq!(result["metric"], "bray_curtis");
    let d = result["result"].as_f64().unwrap();
    assert!(d > 0.0 && d < 1.0, "partial overlap: 0 < BC < 1, got {d}");
}

// ── stats.hill ───────────────────────────────────────────────────────────

#[test]
fn test_stats_hill_half_max() {
    let resp = stats_hill(
        &serde_json::json!({"x": 10.0, "k": 10.0, "n": 2.0}),
        serde_json::json!(20),
    );
    let result = resp.result.expect("hill should succeed");
    let h = result["result"].as_f64().unwrap();
    assert!((h - 0.5).abs() < 1e-12, "x=k → Hill = 0.5, got {h}");
}

#[test]
fn test_stats_hill_saturation() {
    let resp = stats_hill(
        &serde_json::json!({"x": 1000.0, "k": 1.0, "n": 4.0}),
        serde_json::json!(21),
    );
    let result = resp.result.expect("hill should succeed");
    let h = result["result"].as_f64().unwrap();
    assert!(h > 0.999, "x >> k → Hill ≈ 1, got {h}");
}

#[test]
fn test_stats_hill_missing_params() {
    let resp = stats_hill(
        &serde_json::json!({"x": 5.0, "k": 1.0}),
        serde_json::json!(22),
    );
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_dispatch_stats_hill() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.hill",
        &serde_json::json!({"x": 5.0, "k": 2.0, "n": 3.0}),
        serde_json::json!(23),
    )
    .await;
    let result = resp.result.expect("dispatch should succeed");
    assert_eq!(result["function"], "hill");
}
