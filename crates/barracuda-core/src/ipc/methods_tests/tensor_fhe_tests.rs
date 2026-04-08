// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for tensor and FHE error paths.

use super::*;

// ── tensor error paths ──────────────────────────────────────────────────

#[tokio::test]
async fn test_tensor_create_no_gpu() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": [2, 3]}),
        serde_json::json!(13),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_tensor_create_missing_shape() {
    let primal = test_primal();
    let resp = tensor_create(&primal, &serde_json::json!({}), serde_json::json!(14)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_tensor_matmul_tensors_not_found() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a", "rhs_id": "b"}),
        serde_json::json!(15),
    )
    .await;
    let err = resp.error.expect("nonexistent tensors return error");
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("Tensor not found"));
}

#[tokio::test]
async fn test_tensor_matmul_missing_lhs() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"rhs_id": "b"}),
        serde_json::json!(160),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_tensor_matmul_missing_rhs() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a"}),
        serde_json::json!(161),
    )
    .await;
    assert!(resp.error.is_some());
}

// ── FHE error paths ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_fhe_ntt_no_gpu() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(17),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_fhe_ntt_missing_modulus() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"degree": 4, "root_of_unity": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(170),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_degree() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "root_of_unity": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(171),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_root_of_unity() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(172),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_coefficients() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4}),
        serde_json::json!(173),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_params() {
    let primal = test_primal();
    let resp = fhe_ntt(&primal, &serde_json::json!({}), serde_json::json!(18)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_pointwise_mul_no_gpu() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}),
        serde_json::json!(19),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_a() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "b": [5, 6, 7, 8]}),
        serde_json::json!(180),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_b() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1, 2, 3, 4]}),
        serde_json::json!(181),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_params() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(&primal, &serde_json::json!({}), serde_json::json!(20)).await;
    assert!(resp.error.is_some());
}
