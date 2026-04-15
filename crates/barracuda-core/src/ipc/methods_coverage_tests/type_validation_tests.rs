// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for JSON type validation: wrong types for IDs, numeric
//! fields, shapes, and batch ops.

use super::test_primal;
use crate::ipc::jsonrpc;

use super::super::{
    batch,
    compute::compute_dispatch,
    fhe::{fhe_ntt, fhe_pointwise_mul},
    math::{activation_hick, math_sigmoid, rng_uniform, stats_mean},
    tensor::{tensor_clamp, tensor_create, tensor_matmul, tensor_scale},
};

// ── non-string IDs ──────────────────────────────────────────────────

#[tokio::test]
async fn tensor_matmul_lhs_id_non_string() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": 42, "rhs_id": "b"}),
        serde_json::json!(700),
    )
    .await;
    let err = resp.error.expect("numeric lhs_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("lhs_id"));
}

#[tokio::test]
async fn tensor_matmul_rhs_id_non_string() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a", "rhs_id": true}),
        serde_json::json!(701),
    )
    .await;
    let err = resp.error.expect("boolean rhs_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("rhs_id"));
}

#[tokio::test]
async fn tensor_create_shape_not_array() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": 4}),
        serde_json::json!(702),
    )
    .await;
    let err = resp.error.expect("scalar shape should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("shape"));
}

#[tokio::test]
async fn compute_dispatch_shape_not_array() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "zeros", "shape": "4,4"}),
        serde_json::json!(703),
    )
    .await;
    let err = resp.error.expect("string shape should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("shape"));
}

#[tokio::test]
async fn compute_dispatch_op_not_string() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": 42, "shape": [2]}),
        serde_json::json!(704),
    )
    .await;
    let err = resp.error.expect("numeric op should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

// ── wrong JSON types for numeric fields ─────────────────────────────

#[tokio::test]
async fn tensor_scale_scalar_string() {
    let primal = test_primal();
    let resp = tensor_scale(
        &primal,
        &serde_json::json!({"tensor_id": "t_missing", "scalar": "two"}),
        serde_json::json!(710),
    )
    .await;
    let err = resp.error.expect("string scalar should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("scalar"));
}

#[tokio::test]
async fn tensor_clamp_min_string() {
    let primal = test_primal();
    let resp = tensor_clamp(
        &primal,
        &serde_json::json!({"tensor_id": "t_missing", "min": "lo", "max": 1.0}),
        serde_json::json!(711),
    )
    .await;
    let err = resp.error.expect("string min should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("min"));
}

// ── FHE type validation ─────────────────────────────────────────────

#[tokio::test]
async fn fhe_ntt_modulus_string() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({
            "coefficients": [1, 2, 3, 4],
            "modulus": "17",
            "root_of_unity": 3,
            "degree": 4
        }),
        serde_json::json!(720),
    )
    .await;
    let err = resp.error.expect("string modulus should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("modulus"));
}

#[tokio::test]
async fn fhe_pointwise_mul_a_not_array() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"a": "not-array", "b": [1, 2], "modulus": 17}),
        serde_json::json!(721),
    )
    .await;
    let err = resp.error.expect("non-array a should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

// ── math/stats type validation ──────────────────────────────────────

#[test]
fn math_sigmoid_data_not_array() {
    let params = serde_json::json!({"data": "not-array"});
    let id = serde_json::json!(730);
    let resp = math_sigmoid(&params, id);
    let err = resp.error.expect("string data should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("data"));
}

#[test]
fn stats_mean_empty_data() {
    let params = serde_json::json!({"data": []});
    let id = serde_json::json!(731);
    let resp = stats_mean(&params, id);
    assert!(resp.error.is_none(), "empty data should succeed with 0.0");
}

#[test]
fn rng_uniform_n_zero() {
    let params = serde_json::json!({"n": 0, "min": 0.0, "max": 1.0});
    let id = serde_json::json!(732);
    let resp = rng_uniform(&params, id);
    assert!(resp.error.is_none(), "n=0 should succeed with empty array");
}

#[test]
fn activation_hick_n_choices_float() {
    let params = serde_json::json!({"n_choices": 4.5});
    let id = serde_json::json!(733);
    let resp = activation_hick(&params, id);
    let err = resp.error.expect("float n_choices should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("n_choices"));
}

// ── batch edge cases ────────────────────────────────────────────────

#[tokio::test]
async fn batch_ops_not_array() {
    let primal = test_primal();
    let resp = batch::tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": "not-an-array"}),
        serde_json::json!(740),
    )
    .await;
    let err = resp.error.expect("non-array ops should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("ops"));
}
