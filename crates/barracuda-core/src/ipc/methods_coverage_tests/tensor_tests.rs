// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for tensor.* validation paths (create, matmul, add, scale,
//! clamp, reduce, sigmoid).

use super::test_primal;
use crate::ipc::jsonrpc;

use super::super::tensor::{
    tensor_add, tensor_clamp, tensor_create, tensor_matmul, tensor_reduce, tensor_scale,
    tensor_sigmoid,
};

// ── tensor.create ───────────────────────────────────────────────────

#[tokio::test]
async fn tensor_create_missing_shape() {
    let primal = test_primal();
    let resp = tensor_create(&primal, &serde_json::json!({}), serde_json::json!(10)).await;
    let err = resp.error.expect("missing shape should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("shape"));
}

#[tokio::test]
async fn tensor_create_shape_with_non_numeric_filtered() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": ["bad"]}),
        serde_json::json!(11),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn tensor_create_data_length_mismatch() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": [2, 3], "data": [1.0, 2.0]}),
        serde_json::json!(12),
    )
    .await;
    let err = resp.error.expect("data length mismatch should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("does not match"));
}

#[tokio::test]
async fn tensor_create_valid_params_no_gpu() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": [2, 2], "data": [1.0, 2.0, 3.0, 4.0]}),
        serde_json::json!(13),
    )
    .await;
    let err = resp.error.expect("valid params + no GPU = internal error");
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
}

// ── tensor.matmul ───────────────────────────────────────────────────

#[tokio::test]
async fn tensor_matmul_missing_lhs_id() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"rhs_id": "b"}),
        serde_json::json!(20),
    )
    .await;
    let err = resp.error.expect("missing lhs_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn tensor_matmul_missing_rhs_id() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a"}),
        serde_json::json!(21),
    )
    .await;
    let err = resp.error.expect("missing rhs_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn tensor_matmul_lhs_not_found() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "nonexistent", "rhs_id": "also_nonexistent"}),
        serde_json::json!(22),
    )
    .await;
    let err = resp.error.expect("nonexistent lhs should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("nonexistent"));
}

// ── tensor.add ──────────────────────────────────────────────────────

#[tokio::test]
async fn tensor_add_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_add(&primal, &serde_json::json!({}), serde_json::json!(200)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn tensor_add_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_add(
        &primal,
        &serde_json::json!({"tensor_id": "nonexistent"}),
        serde_json::json!(201),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn tensor_add_scalar_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_add(
        &primal,
        &serde_json::json!({"tensor_id": "t_fake", "scalar": 1.0}),
        serde_json::json!(202),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn tensor_add_other_id_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_add(
        &primal,
        &serde_json::json!({"tensor_id": "t_fake", "other_id": "t_missing"}),
        serde_json::json!(203),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.scale ────────────────────────────────────────────────────

#[tokio::test]
async fn tensor_scale_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_scale(&primal, &serde_json::json!({}), serde_json::json!(210)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn tensor_scale_missing_scalar() {
    let primal = test_primal();
    let resp = tensor_scale(
        &primal,
        &serde_json::json!({"tensor_id": "t_fake"}),
        serde_json::json!(211),
    )
    .await;
    let err = resp.error.expect("missing scalar should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("scalar"));
}

#[tokio::test]
async fn tensor_scale_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_scale(
        &primal,
        &serde_json::json!({"tensor_id": "t_none", "scalar": 2.0}),
        serde_json::json!(212),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.clamp ────────────────────────────────────────────────────

#[tokio::test]
async fn tensor_clamp_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_clamp(&primal, &serde_json::json!({}), serde_json::json!(220)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn tensor_clamp_missing_min() {
    let primal = test_primal();
    let resp = tensor_clamp(
        &primal,
        &serde_json::json!({"tensor_id": "t_fake", "max": 1.0}),
        serde_json::json!(221),
    )
    .await;
    let err = resp.error.expect("missing min should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("min"));
}

#[tokio::test]
async fn tensor_clamp_missing_max() {
    let primal = test_primal();
    let resp = tensor_clamp(
        &primal,
        &serde_json::json!({"tensor_id": "t_fake", "min": 0.0}),
        serde_json::json!(222),
    )
    .await;
    let err = resp.error.expect("missing max should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("max"));
}

#[tokio::test]
async fn tensor_clamp_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_clamp(
        &primal,
        &serde_json::json!({"tensor_id": "t_none", "min": 0.0, "max": 1.0}),
        serde_json::json!(223),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.reduce ───────────────────────────────────────────────────

#[tokio::test]
async fn tensor_reduce_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_reduce(&primal, &serde_json::json!({}), serde_json::json!(230)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn tensor_reduce_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_reduce(
        &primal,
        &serde_json::json!({"tensor_id": "t_gone"}),
        serde_json::json!(231),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.sigmoid ──────────────────────────────────────────────────

#[tokio::test]
async fn tensor_sigmoid_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_sigmoid(&primal, &serde_json::json!({}), serde_json::json!(240)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("tensor_id"));
}

#[tokio::test]
async fn tensor_sigmoid_tensor_not_found() {
    let primal = test_primal();
    let resp = tensor_sigmoid(
        &primal,
        &serde_json::json!({"tensor_id": "t_missing"}),
        serde_json::json!(241),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}
