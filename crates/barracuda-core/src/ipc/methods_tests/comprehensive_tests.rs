// SPDX-License-Identifier: AGPL-3.0-or-later
//! Comprehensive tests: all routes, tolerances, tensor store, and full dispatch.

use super::*;

// ── all routes ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_all_dispatch_routes_exist() {
    let primal = test_primal();
    for method in REGISTERED_METHODS {
        let resp = dispatch(
            &primal,
            method,
            &serde_json::json!({}),
            serde_json::json!(99),
        )
        .await;
        if let Some(err) = &resp.error {
            assert_ne!(err.code, METHOD_NOT_FOUND, "Method {method} not routed");
        }
    }
}

// ── tolerances comprehensive ────────────────────────────────────────────

#[test]
#[expect(clippy::float_cmp, reason = "exact tolerance comparison in test")]
fn test_tolerances_all_precisions() {
    for (name, abs_tol) in [("fhe", 0.0), ("f64", 1e-12), ("f32", 1e-5), ("df64", 1e-10)] {
        let resp = tolerances_get(&serde_json::json!({"name": name}), serde_json::json!(name));
        let result = resp.result.unwrap();
        assert_eq!(result["abs_tol"].as_f64().unwrap(), abs_tol);
    }
}

// ── tensor store ────────────────────────────────────────────────────────

#[test]
fn test_tensor_store() {
    let primal = test_primal();
    assert_eq!(primal.tensor_count(), 0);
    assert!(primal.get_tensor("nonexistent").is_none());
}

// ── dispatch via JSON-RPC text protocol (all routes) ────────────────────

#[tokio::test]
async fn test_dispatch_device_probe() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "device.probe",
        &serde_json::json!({}),
        serde_json::json!(200),
    )
    .await;
    let result = resp.result.expect("device.probe always succeeds");
    assert_eq!(result["available"], false);
}

#[tokio::test]
async fn test_dispatch_health_check() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health.check",
        &serde_json::json!({}),
        serde_json::json!(201),
    )
    .await;
    let result = resp.result.expect("health.check succeeds without GPU");
    assert_eq!(result["name"], "barraCuda");
}

#[tokio::test]
async fn test_dispatch_tolerances_get() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tolerances.get",
        &serde_json::json!({"name": "f32"}),
        serde_json::json!(202),
    )
    .await;
    let result = resp.result.expect("tolerances.get always succeeds");
    assert_eq!(result["name"], "f32");
}

#[tokio::test]
async fn test_dispatch_validate_gpu_stack() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "validate.gpu_stack",
        &serde_json::json!({}),
        serde_json::json!(203),
    )
    .await;
    assert!(resp.error.is_some(), "validate without GPU returns error");
}

#[tokio::test]
async fn test_dispatch_compute_dispatch() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "compute.dispatch",
        &serde_json::json!({"op": "zeros", "shape": [4]}),
        serde_json::json!(204),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_tensor_create() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tensor.create",
        &serde_json::json!({"shape": [2, 2]}),
        serde_json::json!(205),
    )
    .await;
    assert!(
        resp.error.is_none(),
        "CPU fallback should succeed without GPU"
    );
    let result = resp.result.unwrap();
    assert_eq!(result["shape"], serde_json::json!([2, 2]));
    assert_eq!(result["backend"], "cpu");
}

#[tokio::test]
async fn test_dispatch_tensor_matmul() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tensor.matmul",
        &serde_json::json!({"lhs_id": "a", "rhs_id": "b"}),
        serde_json::json!(206),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_fhe_ntt() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "fhe.ntt",
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4, "coefficients": [1,2,3,4]}),
        serde_json::json!(207),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_fhe_pointwise_mul() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "fhe.pointwise_mul",
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1,2,3,4], "b": [5,6,7,8]}),
        serde_json::json!(208),
    )
    .await;
    assert!(resp.error.is_some());
}
