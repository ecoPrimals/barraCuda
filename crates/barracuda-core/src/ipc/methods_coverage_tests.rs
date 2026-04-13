// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage-focused tests for JSON-RPC method handlers.
//!
//! These tests exercise validation error paths that are reachable because
//! handlers validate input parameters *before* checking device availability.
//! This makes every validation branch testable without GPU hardware.

use super::compute::{compute_dispatch, parse_shape};
use super::fhe::{fhe_ntt, fhe_pointwise_mul};
use super::health::health_readiness;
use super::math::{
    activation_fitts, activation_hick, math_log2, math_sigmoid, noise_perlin2d, noise_perlin3d,
    rng_uniform, stats_mean, stats_std_dev, stats_weighted_mean,
};
use super::tensor::{
    tensor_add, tensor_clamp, tensor_create, tensor_matmul, tensor_reduce, tensor_scale,
    tensor_sigmoid,
};
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

// ── math.sigmoid validation + happy paths ────────────────────────────

#[test]
fn math_sigmoid_missing_data() {
    let resp = math_sigmoid(&serde_json::json!({}), serde_json::json!(100));
    let err = resp.error.expect("missing data should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("data"));
}

#[test]
fn math_sigmoid_happy_path() {
    let resp = math_sigmoid(
        &serde_json::json!({"data": [0.0, 1.0, -1.0]}),
        serde_json::json!(101),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let arr = result["result"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert!((arr[0].as_f64().unwrap() - 0.5).abs() < 1e-10);
}

// ── math.log2 ────────────────────────────────────────────────────────

#[test]
fn math_log2_missing_data() {
    let resp = math_log2(&serde_json::json!({}), serde_json::json!(102));
    assert!(resp.error.is_some());
}

#[test]
fn math_log2_happy_path() {
    let resp = math_log2(
        &serde_json::json!({"data": [1.0, 2.0, 8.0]}),
        serde_json::json!(103),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert!((arr[0].as_f64().unwrap()).abs() < 1e-10);
    assert!((arr[1].as_f64().unwrap() - 1.0).abs() < 1e-10);
    assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-10);
}

// ── stats.mean ───────────────────────────────────────────────────────

#[test]
fn stats_mean_missing_data() {
    let resp = stats_mean(&serde_json::json!({}), serde_json::json!(104));
    assert!(resp.error.is_some());
}

#[test]
fn stats_mean_happy_path() {
    let resp = stats_mean(
        &serde_json::json!({"data": [2.0, 4.0, 6.0]}),
        serde_json::json!(105),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((result - 4.0).abs() < 1e-10);
}

// ── stats.std_dev ────────────────────────────────────────────────────

#[test]
fn stats_std_dev_missing_data() {
    let resp = stats_std_dev(&serde_json::json!({}), serde_json::json!(106));
    assert!(resp.error.is_some());
}

#[test]
fn stats_std_dev_happy_path() {
    let resp = stats_std_dev(
        &serde_json::json!({"data": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}),
        serde_json::json!(107),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!(result > 0.0);
}

// ── stats.weighted_mean ──────────────────────────────────────────────

#[test]
fn stats_weighted_mean_missing_values() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"weights": [1.0]}),
        serde_json::json!(108),
    );
    assert!(resp.error.is_some());
}

#[test]
fn stats_weighted_mean_missing_weights() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [1.0]}),
        serde_json::json!(109),
    );
    assert!(resp.error.is_some());
}

#[test]
fn stats_weighted_mean_length_mismatch() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [1.0, 2.0], "weights": [1.0]}),
        serde_json::json!(110),
    );
    let err = resp.error.expect("length mismatch");
    assert!(err.message.contains("!="));
}

#[test]
fn stats_weighted_mean_zero_weight() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [1.0], "weights": [0.0]}),
        serde_json::json!(111),
    );
    let err = resp.error.expect("zero weight");
    assert!(err.message.contains("zero"));
}

#[test]
fn stats_weighted_mean_happy_path() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [10.0, 20.0], "weights": [1.0, 3.0]}),
        serde_json::json!(112),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((result - 17.5).abs() < 1e-10);
}

// ── noise.perlin2d ───────────────────────────────────────────────────

#[test]
fn noise_perlin2d_missing_x() {
    let resp = noise_perlin2d(&serde_json::json!({"y": 1.0}), serde_json::json!(113));
    assert!(resp.error.is_some());
}

#[test]
fn noise_perlin2d_missing_y() {
    let resp = noise_perlin2d(&serde_json::json!({"x": 1.0}), serde_json::json!(114));
    assert!(resp.error.is_some());
}

#[test]
fn noise_perlin2d_happy_path() {
    let resp = noise_perlin2d(
        &serde_json::json!({"x": 0.5, "y": 0.5}),
        serde_json::json!(115),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!(result.is_finite());
}

// ── noise.perlin3d ───────────────────────────────────────────────────

#[test]
fn noise_perlin3d_missing_z() {
    let resp = noise_perlin3d(
        &serde_json::json!({"x": 1.0, "y": 1.0}),
        serde_json::json!(116),
    );
    assert!(resp.error.is_some());
}

#[test]
fn noise_perlin3d_happy_path() {
    let resp = noise_perlin3d(
        &serde_json::json!({"x": 0.5, "y": 0.5, "z": 0.5}),
        serde_json::json!(117),
    );
    assert!(resp.error.is_none());
}

// ── rng.uniform ──────────────────────────────────────────────────────

#[test]
fn rng_uniform_max_lte_min() {
    let resp = rng_uniform(
        &serde_json::json!({"min": 5.0, "max": 3.0}),
        serde_json::json!(118),
    );
    let err = resp.error.expect("max <= min should fail");
    assert!(err.message.contains("max must be > min"));
}

#[test]
fn rng_uniform_happy_path() {
    let resp = rng_uniform(
        &serde_json::json!({"n": 5, "min": 0.0, "max": 1.0, "seed": 42}),
        serde_json::json!(119),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 5);
    for v in &arr {
        let f = v.as_f64().unwrap();
        assert!((0.0..1.0).contains(&f));
    }
}

#[test]
fn rng_uniform_defaults() {
    let resp = rng_uniform(&serde_json::json!({}), serde_json::json!(120));
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 1);
}

// ── activation.fitts ─────────────────────────────────────────────────

#[test]
fn activation_fitts_missing_distance() {
    let resp = activation_fitts(&serde_json::json!({"width": 10.0}), serde_json::json!(121));
    assert!(resp.error.is_some());
}

#[test]
fn activation_fitts_missing_width() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0}),
        serde_json::json!(122),
    );
    assert!(resp.error.is_some());
}

#[test]
fn activation_fitts_zero_width() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 0.0}),
        serde_json::json!(123),
    );
    let err = resp.error.expect("zero width should fail");
    assert!(err.message.contains("width must be > 0"));
}

#[test]
fn activation_fitts_unknown_variant() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 10.0, "variant": "unknown"}),
        serde_json::json!(124),
    );
    let err = resp.error.expect("unknown variant should fail");
    assert!(err.message.contains("Unknown variant"));
}

#[test]
fn activation_fitts_shannon_happy_path() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 10.0}),
        serde_json::json!(125),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert!(r["movement_time"].as_f64().unwrap() > 0.0);
    assert_eq!(r["variant"], "shannon");
}

#[test]
fn activation_fitts_original_variant() {
    let resp = activation_fitts(
        &serde_json::json!({"distance": 100.0, "width": 10.0, "variant": "fitts"}),
        serde_json::json!(126),
    );
    assert!(resp.error.is_none());
    assert_eq!(resp.result.unwrap()["variant"], "fitts");
}

// ── activation.hick ──────────────────────────────────────────────────

#[test]
fn activation_hick_missing_n_choices() {
    let resp = activation_hick(&serde_json::json!({}), serde_json::json!(127));
    assert!(resp.error.is_some());
}

#[test]
fn activation_hick_zero_choices() {
    let resp = activation_hick(&serde_json::json!({"n_choices": 0}), serde_json::json!(128));
    let err = resp.error.expect("zero choices should fail");
    assert!(err.message.contains("n_choices must be > 0"));
}

#[test]
fn activation_hick_happy_path() {
    let resp = activation_hick(&serde_json::json!({"n_choices": 8}), serde_json::json!(129));
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let bits = r["information_bits"].as_f64().unwrap();
    assert!((bits - 3.0).abs() < 1e-10);
}

#[test]
fn activation_hick_with_no_choice() {
    let resp = activation_hick(
        &serde_json::json!({"n_choices": 4, "include_no_choice": true}),
        serde_json::json!(130),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert!(r["include_no_choice"].as_bool().unwrap());
    let bits = r["information_bits"].as_f64().unwrap();
    assert!((bits - 5.0_f64.log2()).abs() < 1e-10);
}

// ── tensor.add validation paths ─────────────────────────────────────

#[tokio::test]
async fn tensor_add_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_add(&primal, &serde_json::json!({}), serde_json::json!(200)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.scale validation paths ───────────────────────────────────

#[tokio::test]
async fn tensor_scale_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_scale(&primal, &serde_json::json!({}), serde_json::json!(210)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.clamp validation paths ───────────────────────────────────

#[tokio::test]
async fn tensor_clamp_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_clamp(&primal, &serde_json::json!({}), serde_json::json!(220)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.reduce validation paths ──────────────────────────────────

#[tokio::test]
async fn tensor_reduce_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_reduce(&primal, &serde_json::json!({}), serde_json::json!(230)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── tensor.sigmoid validation paths ─────────────────────────────────

#[tokio::test]
async fn tensor_sigmoid_missing_tensor_id() {
    let primal = test_primal();
    let resp = tensor_sigmoid(&primal, &serde_json::json!({}), serde_json::json!(240)).await;
    let err = resp.error.expect("missing tensor_id should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("not found"));
}

// ── FHE degree overflow validation (JSON-RPC path) ──────────────────

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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("too large") || err.message.contains("u32::MAX"));
}

// ── stats.std_dev edge cases ────────────────────────────────────────

#[test]
fn stats_std_dev_empty_data() {
    let resp = stats_std_dev(&serde_json::json!({"data": []}), serde_json::json!(600));
    let err = resp.error.expect("empty data should fail");
    assert_eq!(err.code, super::super::jsonrpc::INTERNAL_ERROR);
    assert!(err.message.contains("std_dev failed"));
}

#[test]
fn stats_std_dev_single_element() {
    let resp = stats_std_dev(&serde_json::json!({"data": [42.0]}), serde_json::json!(601));
    let err = resp
        .error
        .expect("single element should fail for sample std_dev");
    assert_eq!(err.code, super::super::jsonrpc::INTERNAL_ERROR);
    assert!(err.message.contains("std_dev failed"));
}

// ── noise.perlin3d missing individual params ────────────────────────

#[test]
fn noise_perlin3d_missing_x_only() {
    let resp = noise_perlin3d(
        &serde_json::json!({"y": 1.0, "z": 2.0}),
        serde_json::json!(602),
    );
    let err = resp.error.expect("missing x should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("x"));
}

#[test]
fn noise_perlin3d_missing_y_only() {
    let resp = noise_perlin3d(
        &serde_json::json!({"x": 1.0, "z": 2.0}),
        serde_json::json!(603),
    );
    let err = resp.error.expect("missing y should fail");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("y"));
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
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
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Unknown op"));
}
