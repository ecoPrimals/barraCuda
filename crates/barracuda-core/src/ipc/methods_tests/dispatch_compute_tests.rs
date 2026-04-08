// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for dispatch routing, validate, and compute.dispatch error paths.

use super::*;

// ── dispatch routing ────────────────────────────────────────────────────

#[tokio::test]
async fn test_dispatch_routing() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "device.list",
        &serde_json::json!({}),
        serde_json::json!(5),
    )
    .await;
    assert!(resp.result.is_some());

    let resp = dispatch(
        &primal,
        "nonexistent.method",
        &serde_json::json!({}),
        serde_json::json!(6),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
}

#[tokio::test]
async fn test_dispatch_wrong_namespace() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "other_primal.device.list",
        &serde_json::json!({}),
        serde_json::json!(140),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
}

// ── validate ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_validate_no_gpu() {
    let primal = test_primal();
    let resp = validate_gpu_stack(&primal, serde_json::json!(10)).await;
    assert!(resp.error.is_some());
    let err = resp.error.unwrap();
    assert_eq!(err.code, INTERNAL_ERROR);
}

// ── compute.dispatch error paths ────────────────────────────────────────

#[tokio::test]
async fn test_compute_dispatch_no_gpu() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "zeros", "shape": [4]}),
        serde_json::json!(11),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_compute_dispatch_missing_op() {
    let primal = test_primal();
    let resp = compute_dispatch(&primal, &serde_json::json!({}), serde_json::json!(12)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_compute_dispatch_ones_no_gpu() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "ones", "shape": [2, 3]}),
        serde_json::json!(150),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_compute_dispatch_read_nonexistent_tensor() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "read", "tensor_id": "nonexistent"}),
        serde_json::json!(151),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor returns error");
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("Tensor not found"));
}

#[tokio::test]
async fn test_compute_dispatch_unknown_op_no_gpu() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "unknown_operation"}),
        serde_json::json!(152),
    )
    .await;
    assert!(resp.error.is_some());
}
