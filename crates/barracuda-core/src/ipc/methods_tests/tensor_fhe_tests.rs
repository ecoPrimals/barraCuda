// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for tensor and FHE error paths.

use super::*;

// ── tensor error paths ──────────────────────────────────────────────────

#[tokio::test]
async fn test_tensor_create_cpu_fallback() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": [2, 3]}),
        serde_json::json!(13),
    )
    .await;
    assert!(
        resp.error.is_none(),
        "CPU fallback should succeed without GPU"
    );
    let result = resp.result.unwrap();
    assert_eq!(result["shape"], serde_json::json!([2, 3]));
    assert_eq!(result["elements"], 6);
    assert_eq!(result["backend"], "cpu");
    assert!(result["tensor_id"].is_string());
}

#[tokio::test]
async fn test_cpu_tensor_roundtrip_create_matmul_reduce() {
    let primal = test_primal();

    let lhs = tensor_create(
        &primal,
        &serde_json::json!({"shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}),
        serde_json::json!(100),
    )
    .await;
    let lhs_id = lhs.result.unwrap()["tensor_id"]
        .as_str()
        .unwrap()
        .to_string();

    let rhs = tensor_create(
        &primal,
        &serde_json::json!({"shape": [3, 2], "data": [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}),
        serde_json::json!(101),
    )
    .await;
    let rhs_id = rhs.result.unwrap()["tensor_id"]
        .as_str()
        .unwrap()
        .to_string();

    let product = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": lhs_id, "rhs_id": rhs_id}),
        serde_json::json!(102),
    )
    .await;
    let prod = product.result.unwrap();
    assert_eq!(prod["shape"], serde_json::json!([2, 2]));

    let prod_id = prod["result_id"].as_str().unwrap();
    let reduced = tensor_reduce(
        &primal,
        &serde_json::json!({"tensor_id": prod_id, "op": "sum"}),
        serde_json::json!(103),
    )
    .await;
    let val = reduced.result.unwrap()["value"].as_f64().unwrap();
    // [1,2,3;4,5,6] * [7,8;9,10;11,12] = [58,64;139,154], sum = 415
    assert!((val - 415.0).abs() < 1e-3, "expected 415, got {val}");
}

#[tokio::test]
async fn test_cpu_tensor_add_scale_clamp_sigmoid() {
    let primal = test_primal();

    let t = tensor_create(
        &primal,
        &serde_json::json!({"shape": [1, 4], "data": [-2.0, -1.0, 0.0, 1.0]}),
        serde_json::json!(200),
    )
    .await;
    let tid = t.result.unwrap()["tensor_id"].as_str().unwrap().to_string();

    let scaled = tensor_scale(
        &primal,
        &serde_json::json!({"tensor_id": tid, "scalar": 2.0}),
        serde_json::json!(201),
    )
    .await;
    assert!(scaled.error.is_none());

    let added = tensor_add(
        &primal,
        &serde_json::json!({"tensor_id": tid, "scalar": 10.0}),
        serde_json::json!(202),
    )
    .await;
    assert!(added.error.is_none());

    let clamped = tensor_clamp(
        &primal,
        &serde_json::json!({"tensor_id": tid, "min": -1.0, "max": 0.5}),
        serde_json::json!(203),
    )
    .await;
    assert!(clamped.error.is_none());

    let sig = tensor_sigmoid(
        &primal,
        &serde_json::json!({"tensor_id": tid}),
        serde_json::json!(204),
    )
    .await;
    assert!(sig.error.is_none());
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
