// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wave 53 coverage expansion — fills handler gaps identified by primalSpring audit.
//!
//! Targets:
//! - `stats.variance` (previously zero dedicated tests)
//! - `ml.esn_predict` happy path (previously validation-only)
//! - `ml.mlp_train` happy path (previously zero tests)
//! - `auth.*` dispatch integration (previously unit-tested in gate only)
//! - `stats.correlation` error paths

use crate::BarraCudaPrimal;
use crate::ipc::jsonrpc;
use crate::ipc::methods::dispatch;

use super::super::ml::{ml_esn_predict, ml_mlp_train};
use super::super::stats::{stats_correlation, stats_variance};

// ── stats.variance ──────────────────────────────────────────────────

#[test]
fn stats_variance_missing_data() {
    let resp = stats_variance(&serde_json::json!({}), serde_json::json!(5300));
    let err = resp.error.expect("missing data should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("data"));
}

#[test]
fn stats_variance_happy_path() {
    let resp = stats_variance(
        &serde_json::json!({"data": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}),
        serde_json::json!(5301),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let variance = result["result"].as_f64().unwrap();
    assert!((variance - 4.571_428_571_428_571).abs() < 1e-10);
    assert_eq!(result["convention"], "sample");
    assert_eq!(result["denominator"], "N-1");
}

#[test]
fn stats_variance_single_element() {
    let resp = stats_variance(&serde_json::json!({"data": [5.0]}), serde_json::json!(5302));
    let err = resp.error.expect("single element should fail (N-1=0)");
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
}

#[test]
fn stats_variance_two_elements() {
    let resp = stats_variance(
        &serde_json::json!({"data": [3.0, 7.0]}),
        serde_json::json!(5303),
    );
    assert!(resp.error.is_none());
    let variance = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((variance - 8.0).abs() < 1e-10);
}

#[test]
fn stats_variance_identical_values() {
    let resp = stats_variance(
        &serde_json::json!({"data": [3.0, 3.0, 3.0, 3.0]}),
        serde_json::json!(5304),
    );
    assert!(resp.error.is_none());
    let variance = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!(variance.abs() < 1e-15);
}

// ── stats.correlation error paths ───────────────────────────────────

#[test]
fn stats_correlation_missing_x() {
    let resp = stats_correlation(
        &serde_json::json!({"y": [1.0, 2.0]}),
        serde_json::json!(5310),
    );
    let err = resp.error.expect("missing x");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_correlation_missing_y() {
    let resp = stats_correlation(
        &serde_json::json!({"x": [1.0, 2.0]}),
        serde_json::json!(5311),
    );
    let err = resp.error.expect("missing y");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn stats_correlation_length_mismatch() {
    let resp = stats_correlation(
        &serde_json::json!({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0]}),
        serde_json::json!(5312),
    );
    let err = resp.error.expect("length mismatch");
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
}

// ── ml.esn_predict happy path ───────────────────────────────────────

#[test]
fn ml_esn_predict_happy_path() {
    use barracuda::nn::esn_classifier::{EsnClassifier, EsnConfig};

    let config = EsnConfig {
        input_size: 2,
        reservoir_size: 10,
        output_size: 1,
        spectral_radius: 0.9,
        sparsity: 0.3,
        leak_rate: 0.5,
        regularization: 1e-4,
        seed: 42,
    };
    let mut esn = EsnClassifier::new(config).unwrap();

    let inputs: Vec<Vec<f64>> = (0..20_i32)
        .map(|i| vec![f64::from(i) * 0.1, (f64::from(i) * 0.2).sin()])
        .collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0] + x[1]]).collect();
    esn.train(&inputs, &targets).unwrap();

    let weights_json = esn.to_json().unwrap();

    let resp = ml_esn_predict(
        &serde_json::json!({
            "weights_json": weights_json,
            "input": [0.5, 0.3]
        }),
        serde_json::json!(5320),
    );
    assert!(
        resp.error.is_none(),
        "ESN predict should succeed: {:?}",
        resp.error
    );
    let result = resp.result.unwrap();
    let prediction = result["prediction"].as_array().unwrap();
    assert_eq!(prediction.len(), 1);
    assert!(prediction[0].as_f64().unwrap().is_finite());
    let state = result["state"].as_array().unwrap();
    assert_eq!(state.len(), 10);
}

#[test]
fn ml_esn_predict_with_state_injection() {
    use barracuda::nn::esn_classifier::{EsnClassifier, EsnConfig};

    let config = EsnConfig {
        input_size: 1,
        reservoir_size: 5,
        output_size: 1,
        spectral_radius: 0.9,
        sparsity: 0.5,
        leak_rate: 0.3,
        regularization: 1e-4,
        seed: 7,
    };
    let mut esn = EsnClassifier::new(config).unwrap();
    let inputs: Vec<Vec<f64>> = (0..30_i32).map(|i| vec![f64::from(i) * 0.05]).collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0] * 2.0]).collect();
    esn.train(&inputs, &targets).unwrap();

    let weights_json = esn.to_json().unwrap();
    let initial_state = vec![0.1, -0.1, 0.2, -0.2, 0.0];

    let resp = ml_esn_predict(
        &serde_json::json!({
            "weights_json": weights_json,
            "input": [0.7],
            "state": initial_state
        }),
        serde_json::json!(5321),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result["prediction"][0].as_f64().unwrap().is_finite());
}

#[test]
fn ml_esn_predict_state_wrong_size() {
    use barracuda::nn::esn_classifier::{EsnClassifier, EsnConfig};

    let config = EsnConfig {
        input_size: 1,
        reservoir_size: 5,
        output_size: 1,
        ..EsnConfig::default()
    };
    let mut esn = EsnClassifier::new(config).unwrap();
    let inputs: Vec<Vec<f64>> = (0..10_i32).map(|i| vec![f64::from(i)]).collect();
    let targets: Vec<Vec<f64>> = inputs.iter().map(|x| vec![x[0]]).collect();
    esn.train(&inputs, &targets).unwrap();

    let weights_json = esn.to_json().unwrap();

    let resp = ml_esn_predict(
        &serde_json::json!({
            "weights_json": weights_json,
            "input": [1.0],
            "state": [0.1, 0.2, 0.3]
        }),
        serde_json::json!(5322),
    );
    let err = resp.error.expect("state size mismatch");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("state length"));
}

// ── ml.mlp_train happy path ─────────────────────────────────────────

#[test]
fn ml_mlp_train_happy_path() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [
                {"weights": [[0.5, 0.5], [0.5, 0.5]], "biases": [0.0, 0.0], "activation": "relu"},
                {"weights": [[1.0, 1.0]], "biases": [0.0], "activation": "identity"}
            ],
            "inputs": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            "targets": [[0.0], [1.0], [1.0], [1.0]],
            "epochs": 100,
            "learning_rate": 0.01
        }),
        serde_json::json!(5330),
    );
    assert!(
        resp.error.is_none(),
        "mlp_train should succeed: {:?}",
        resp.error
    );
    let result = resp.result.unwrap();
    let layers = result["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 2);
    assert!(result["mse"].as_f64().unwrap().is_finite());
    assert_eq!(result["epochs"].as_u64().unwrap(), 100);
}

#[test]
fn ml_mlp_train_missing_inputs() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [{"weights": [[1.0]], "biases": [0.0], "activation": "relu"}],
            "targets": [[1.0]],
            "epochs": 10,
            "learning_rate": 0.01
        }),
        serde_json::json!(5331),
    );
    let err = resp.error.expect("missing inputs");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_mlp_train_missing_targets() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [{"weights": [[1.0]], "biases": [0.0], "activation": "relu"}],
            "inputs": [[1.0]],
            "epochs": 10,
            "learning_rate": 0.01
        }),
        serde_json::json!(5332),
    );
    let err = resp.error.expect("missing targets");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

// ── auth.* dispatch integration ─────────────────────────────────────

#[tokio::test]
async fn auth_check_dispatch_no_token() {
    let primal = BarraCudaPrimal::new();
    let resp = dispatch(
        &primal,
        "auth.check",
        &serde_json::json!({}),
        serde_json::json!(5340),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["authenticated"], false);
    assert!(result.get("origin").is_some());
}

#[tokio::test]
async fn auth_mode_dispatch() {
    let primal = BarraCudaPrimal::new();
    let resp = dispatch(
        &primal,
        "auth.mode",
        &serde_json::json!({}),
        serde_json::json!(5341),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result.get("mode").is_some());
    assert_eq!(result["standard"], "METHOD_GATE_STANDARD v1.0");
}

#[tokio::test]
async fn auth_peer_info_dispatch() {
    let primal = BarraCudaPrimal::new();
    let resp = dispatch(
        &primal,
        "auth.peer_info",
        &serde_json::json!({}),
        serde_json::json!(5342),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result.get("origin").is_some());
    assert_eq!(result["peer"]["available"], false);
}

#[tokio::test]
async fn auth_check_dispatch_with_bearer() {
    let primal = BarraCudaPrimal::new();
    let resp = dispatch(
        &primal,
        "auth.check",
        &serde_json::json!({"_auth": {"bearer": "test-token-abc"}}),
        serde_json::json!(5343),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert_eq!(result["authenticated"], true);
}
