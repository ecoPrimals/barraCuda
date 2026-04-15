// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for FHE method handlers (`fhe.ntt`, `fhe.pointwise_mul`).

use super::test_primal;
use crate::ipc::jsonrpc;

use super::super::fhe::{fhe_ntt, fhe_pointwise_mul};

// ── fhe.ntt validation paths ────────────────────────────────────────

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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
}

// ── fhe.pointwise_mul validation paths ──────────────────────────────

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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
}

// ── FHE degree overflow (JSON-RPC path) ─────────────────────────────

#[tokio::test]
async fn fhe_ntt_degree_exceeds_u32_max() {
    let primal = test_primal();
    let big_degree = u64::from(u32::MAX) + 1;
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({
            "modulus": 17,
            "degree": big_degree,
            "root_of_unity": 3,
            "coefficients": [1, 2, 3]
        }),
        serde_json::json!(500),
    )
    .await;
    let err = resp.error.expect("degree > u32::MAX should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("too large") || err.message.contains("u32::MAX"));
}

#[tokio::test]
async fn fhe_pointwise_mul_degree_exceeds_u32_max() {
    let primal = test_primal();
    let big_degree = u64::from(u32::MAX) + 1;
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({
            "modulus": 17,
            "degree": big_degree,
            "a": [1],
            "b": [2]
        }),
        serde_json::json!(501),
    )
    .await;
    let err = resp.error.expect("degree > u32::MAX should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("too large") || err.message.contains("u32::MAX"));
}
