// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for device.list, device.probe, health endpoints, tolerances, and aliases.

use super::*;

// ── device.list and device.probe ────────────────────────────────────────

#[tokio::test]
async fn test_device_list_no_gpu() {
    let primal = test_primal();
    let resp = device_list(&primal, serde_json::json!(1)).await;
    assert!(resp.result.is_some());
    let result = resp.result.unwrap();
    assert!(result["devices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_device_probe_no_gpu() {
    let primal = test_primal();
    let resp = device_probe(&primal, serde_json::json!(120)).await;
    let result = resp.result.expect("device.probe always returns success");
    assert_eq!(result["available"], false);
    assert!(result["reason"].is_string());
}

// ── health and tolerances ───────────────────────────────────────────────

#[test]
fn test_health_liveness() {
    let resp = health_liveness(serde_json::json!(200));
    let result = resp.result.expect("health.liveness always succeeds");
    assert_eq!(result["status"], "alive");
}

#[test]
fn test_health_readiness_not_started() {
    let primal = test_primal();
    let resp = health_readiness(&primal, serde_json::json!(201));
    let result = resp.result.expect("health.readiness always succeeds");
    assert_eq!(result["status"], "not_ready");
    assert_eq!(result["gpu_available"], false);
}

#[tokio::test]
async fn test_health_check() {
    let primal = test_primal();
    let resp = health_check(&primal, serde_json::json!(2)).await;
    assert!(resp.result.is_some());
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "barraCuda");
}

// ── health alias dispatch tests ─────────────────────────────────────────

#[tokio::test]
async fn test_dispatch_health_liveness() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health.liveness",
        &serde_json::json!({}),
        serde_json::json!(210),
    )
    .await;
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[tokio::test]
async fn test_dispatch_ping_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "ping",
        &serde_json::json!({}),
        serde_json::json!(211),
    )
    .await;
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[tokio::test]
async fn test_dispatch_health_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health",
        &serde_json::json!({}),
        serde_json::json!(212),
    )
    .await;
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[tokio::test]
async fn test_dispatch_health_readiness() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health.readiness",
        &serde_json::json!({}),
        serde_json::json!(213),
    )
    .await;
    let result = resp.result.unwrap();
    assert!(result["status"].is_string());
}

#[tokio::test]
async fn test_dispatch_status_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "status",
        &serde_json::json!({}),
        serde_json::json!(214),
    )
    .await;
    assert!(resp.result.is_some(), "status alias for health.check");
}

#[tokio::test]
async fn test_dispatch_check_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "check",
        &serde_json::json!({}),
        serde_json::json!(215),
    )
    .await;
    assert!(resp.result.is_some(), "check alias for health.check");
}

#[tokio::test]
async fn test_dispatch_capabilities_list() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "capabilities.list",
        &serde_json::json!({}),
        serde_json::json!(216),
    )
    .await;
    assert!(resp.result.is_some(), "capabilities.list canonical");
    assert!(resp.result.unwrap()["methods"].is_array());
}

#[tokio::test]
async fn test_dispatch_capability_list_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "capability.list",
        &serde_json::json!({}),
        serde_json::json!(217),
    )
    .await;
    assert!(resp.result.is_some(), "capability.list alias");
}

#[test]
fn test_tolerances_default() {
    let resp = tolerances_get(&serde_json::json!({}), serde_json::json!(3));
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "default");
}

#[test]
fn test_tolerances_fhe() {
    let resp = tolerances_get(&serde_json::json!({"name": "fhe"}), serde_json::json!(4));
    let result = resp.result.unwrap();
    assert_eq!(result["abs_tol"], 0.0);
    assert_eq!(result["rel_tol"], 0.0);
}

#[test]
fn test_tolerances_double_alias() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "double"}),
        serde_json::json!(130),
    );
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "double");
    assert!(result["abs_tol"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_tolerances_emulated_double_alias() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "emulated_double"}),
        serde_json::json!(131),
    );
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "emulated_double");
}

#[test]
fn test_tolerances_float_alias() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "float"}),
        serde_json::json!(132),
    );
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "float");
}

#[test]
fn test_tolerances_unknown_returns_defaults() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "some_unknown_precision"}),
        serde_json::json!(133),
    );
    let result = resp.result.unwrap();
    assert!(result["abs_tol"].as_f64().unwrap() > 0.0);
    assert!(result["rel_tol"].as_f64().unwrap() > 0.0);
}
