// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for `compute.dispatch` validation paths and edge cases.

use super::super::compute::{compute_dispatch, parse_shape};
use super::test_primal;
use crate::ipc::jsonrpc;

#[tokio::test]
async fn compute_missing_op_returns_invalid_params() {
    let primal = test_primal();
    let resp = compute_dispatch(&primal, &serde_json::json!({}), serde_json::json!(1)).await;
    let err = resp.error.expect("missing op should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("op"));
}

#[tokio::test]
async fn compute_unknown_op_returns_invalid_params() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "nonexistent_operation"}),
        serde_json::json!(2),
    )
    .await;
    let err = resp.error.expect("unknown op should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Unknown op"));
}

#[tokio::test]
async fn compute_zeros_missing_shape() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "zeros"}),
        serde_json::json!(3),
    )
    .await;
    let err = resp.error.expect("missing shape should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("shape"));
}

#[tokio::test]
async fn compute_ones_missing_shape() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "ones"}),
        serde_json::json!(4),
    )
    .await;
    let err = resp.error.expect("missing shape should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn compute_zeros_shape_with_non_numeric_filtered() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "zeros", "shape": ["not_a_number"]}),
        serde_json::json!(5),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn compute_read_missing_tensor_id() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "read"}),
        serde_json::json!(6),
    )
    .await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn compute_read_nonexistent_tensor() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "read", "tensor_id": "does_not_exist"}),
        serde_json::json!(7),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Tensor not found"));
}

// ── parse_shape edge cases ──────────────────────────────────────────

#[test]
fn parse_shape_large_values() {
    let arr = vec![serde_json::json!(u64::MAX)];
    let result = parse_shape(&arr);
    if cfg!(target_pointer_width = "64") {
        assert!(result.is_some());
    }
}

#[test]
fn parse_shape_mixed_types() {
    let arr = vec![serde_json::json!(2), serde_json::json!(null)];
    let shape = parse_shape(&arr);
    assert!(
        shape.is_none() || shape.as_ref().is_some_and(|s| s.len() < 2),
        "null values should be filtered or cause failure"
    );
}

// ── compute.dispatch edge cases ─────────────────────────────────────

#[tokio::test]
async fn compute_dispatch_tensor_id_non_string() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "read", "tensor_id": 12345}),
        serde_json::json!(604),
    )
    .await;
    let err = resp.error.expect("numeric tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn compute_dispatch_empty_op() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": ""}),
        serde_json::json!(605),
    )
    .await;
    let err = resp.error.expect("empty op should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Unknown op"));
}
