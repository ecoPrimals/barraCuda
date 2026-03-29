// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage-focused tests for JSON-RPC method handlers.
//!
//! These tests exercise validation error paths that are reachable because
//! handlers validate input parameters *before* checking device availability.
//! This makes every validation branch testable without GPU hardware.

use super::compute::{compute_dispatch, parse_shape};
use super::fhe::{fhe_ntt, fhe_pointwise_mul};
use super::health::health_readiness;
use super::tensor::{tensor_create, tensor_matmul};
use crate::BarraCudaPrimal;
use crate::lifecycle::PrimalLifecycle;

fn test_primal() -> BarraCudaPrimal {
    BarraCudaPrimal::new()
}

// ── compute.dispatch validation paths ────────────────────────────────

#[tokio::test]
async fn compute_missing_op_returns_invalid_params() {
    let primal = test_primal();
    let resp = compute_dispatch(&primal, &serde_json::json!({}), serde_json::json!(1)).await;
    let err = resp.error.expect("missing op should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    // Non-numeric values are filtered by parse_shape → valid empty shape → device check
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Tensor not found"));
}

// ── tensor.create validation paths ───────────────────────────────────

#[tokio::test]
async fn tensor_create_missing_shape() {
    let primal = test_primal();
    let resp = tensor_create(&primal, &serde_json::json!({}), serde_json::json!(10)).await;
    let err = resp.error.expect("missing shape should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    // Non-numeric values are filtered → valid empty shape → device check
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INTERNAL_ERROR);
}

// ── tensor.matmul validation paths ───────────────────────────────────

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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("nonexistent"));
}

// ── fhe.ntt validation paths ─────────────────────────────────────────

#[tokio::test]
async fn fhe_ntt_missing_modulus() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"degree": 4, "root_of_unity": 4, "coefficients": [1,2,3,4]}),
        serde_json::json!(30),
    )
    .await;
    let err = resp.error.expect("missing modulus should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("modulus"));
}

#[tokio::test]
async fn fhe_ntt_missing_degree() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "root_of_unity": 4, "coefficients": [1,2,3,4]}),
        serde_json::json!(31),
    )
    .await;
    let err = resp.error.expect("missing degree should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("degree"));
}

#[tokio::test]
async fn fhe_ntt_missing_root_of_unity() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "coefficients": [1,2,3,4]}),
        serde_json::json!(32),
    )
    .await;
    let err = resp.error.expect("missing root_of_unity should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("root_of_unity"));
}

#[tokio::test]
async fn fhe_ntt_missing_coefficients() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4}),
        serde_json::json!(33),
    )
    .await;
    let err = resp.error.expect("missing coefficients should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("coefficients"));
}

#[tokio::test]
async fn fhe_ntt_coefficient_length_mismatch() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({
            "modulus": 17, "degree": 4, "root_of_unity": 4,
            "coefficients": [1, 2, 3]
        }),
        serde_json::json!(34),
    )
    .await;
    let err = resp.error.expect("length mismatch should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("!="));
}

#[tokio::test]
async fn fhe_ntt_valid_params_no_gpu() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({
            "modulus": 17, "degree": 4, "root_of_unity": 4,
            "coefficients": [1, 2, 3, 4]
        }),
        serde_json::json!(35),
    )
    .await;
    let err = resp.error.expect("valid params + no GPU = internal error");
    assert_eq!(err.code, super::super::jsonrpc::INTERNAL_ERROR);
}

// ── fhe.pointwise_mul validation paths ───────────────────────────────

#[tokio::test]
async fn fhe_pointwise_mul_missing_modulus() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"degree": 4, "a": [1,2,3,4], "b": [5,6,7,8]}),
        serde_json::json!(40),
    )
    .await;
    let err = resp.error.expect("missing modulus should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn fhe_pointwise_mul_missing_degree() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "a": [1,2,3,4], "b": [5,6,7,8]}),
        serde_json::json!(41),
    )
    .await;
    let err = resp.error.expect("missing degree should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn fhe_pointwise_mul_missing_a() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "b": [5,6,7,8]}),
        serde_json::json!(42),
    )
    .await;
    let err = resp.error.expect("missing a should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn fhe_pointwise_mul_missing_b() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1,2,3,4]}),
        serde_json::json!(43),
    )
    .await;
    let err = resp.error.expect("missing b should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn fhe_pointwise_mul_length_mismatch_a() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({
            "modulus": 17, "degree": 4,
            "a": [1, 2, 3],
            "b": [5, 6, 7, 8]
        }),
        serde_json::json!(44),
    )
    .await;
    let err = resp.error.expect("length mismatch should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("elements"));
}

#[tokio::test]
async fn fhe_pointwise_mul_length_mismatch_b() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({
            "modulus": 17, "degree": 4,
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7]
        }),
        serde_json::json!(45),
    )
    .await;
    let err = resp.error.expect("length mismatch should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
}

#[tokio::test]
async fn fhe_pointwise_mul_valid_params_no_gpu() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({
            "modulus": 17, "degree": 4,
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8]
        }),
        serde_json::json!(46),
    )
    .await;
    let err = resp.error.expect("valid params + no GPU = internal error");
    assert_eq!(err.code, super::super::jsonrpc::INTERNAL_ERROR);
}

// ── health.readiness after start ─────────────────────────────────────

#[tokio::test]
async fn health_readiness_after_start() {
    let mut primal = test_primal();
    primal.start().await.unwrap();
    let resp = health_readiness(&primal, serde_json::json!(50));
    let result = resp.result.expect("health.readiness always succeeds");
    assert_eq!(result["status"], "ready");
}

// ── parse_shape edge cases ───────────────────────────────────────────

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
