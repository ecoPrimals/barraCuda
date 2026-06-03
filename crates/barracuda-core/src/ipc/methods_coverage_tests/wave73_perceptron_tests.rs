// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wave 73 perceptron training tests — validates 36-dim single-layer perceptron
//! for biomeOS L5 Neural API routing.
//!
//! Wire contract (from NEURAL_API_PERCEPTRON_DESIGN.md):
//! ```json
//! { "layers": [36, 16], "inputs": [...], "targets": [...],
//!   "learning_rate": 0.01, "epochs": 10 }
//! ```

use crate::ipc::jsonrpc;

use super::super::ml::ml_mlp_train;

#[test]
fn perceptron_36_16_dims_shorthand() {
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Vec<f64>> = Vec::new();

    for i in 0..32 {
        let mut feature_vec = vec![0.0; 36];
        let domain_idx = i % 16;
        feature_vec[domain_idx] = 1.0;
        feature_vec[32] = (i as f64) / 32.0; // param_size_norm
        feature_vec[33] = 0.5;               // gate_load_norm
        feature_vec[34] = (i as f64) / 100.0; // latency_ewma_norm
        feature_vec[35] = 0.8;               // topology_affinity
        inputs.push(feature_vec);

        let mut target = vec![0.0; 16];
        target[domain_idx] = 1.0;
        targets.push(target);
    }

    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [36, 16],
            "inputs": inputs,
            "targets": targets,
            "learning_rate": 0.01,
            "epochs": 10
        }),
        serde_json::json!(7300),
    );

    assert!(resp.error.is_none(), "36→16 perceptron train failed: {:?}", resp.error);
    let result = resp.result.unwrap();

    let layers = result["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1, "single layer for [36, 16]");

    let weights = layers[0]["weights"].as_array().unwrap();
    assert_eq!(weights.len(), 16, "output dim = 16");
    assert_eq!(weights[0].as_array().unwrap().len(), 36, "input dim = 36");

    let biases = layers[0]["biases"].as_array().unwrap();
    assert_eq!(biases.len(), 16);

    let mse = result["mse"].as_f64().unwrap();
    assert!(mse.is_finite());
    assert!(mse < 1.0, "MSE should decrease after training, got {mse}");
    assert_eq!(result["epochs"].as_u64().unwrap(), 10);
}

#[test]
fn perceptron_36_64_16_two_layer() {
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Vec<f64>> = Vec::new();

    for i in 0..16 {
        let mut feature_vec = vec![0.0; 36];
        feature_vec[i % 32] = 1.0;
        feature_vec[34] = 0.3;
        feature_vec[35] = 0.9;
        inputs.push(feature_vec);

        let mut target = vec![0.0; 16];
        target[i] = 1.0;
        targets.push(target);
    }

    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [36, 64, 16],
            "inputs": inputs,
            "targets": targets,
            "activation": "relu",
            "learning_rate": 0.01,
            "epochs": 20
        }),
        serde_json::json!(7301),
    );

    assert!(resp.error.is_none(), "36→64→16 train failed: {:?}", resp.error);
    let result = resp.result.unwrap();

    let layers = result["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 2, "two layers for [36, 64, 16]");

    let l0_weights = layers[0]["weights"].as_array().unwrap();
    assert_eq!(l0_weights.len(), 64);
    assert_eq!(l0_weights[0].as_array().unwrap().len(), 36);

    let l1_weights = layers[1]["weights"].as_array().unwrap();
    assert_eq!(l1_weights.len(), 16);
    assert_eq!(l1_weights[0].as_array().unwrap().len(), 64);

    assert!(result["mse"].as_f64().unwrap().is_finite());
}

#[test]
fn perceptron_dims_shorthand_default_sigmoid() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [4, 2],
            "inputs": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            "targets": [[1.0, 0.0], [0.0, 1.0]],
            "epochs": 5
        }),
        serde_json::json!(7302),
    );

    assert!(resp.error.is_none(), "default sigmoid shorthand failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    let layers = result["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1);
    assert_eq!(layers[0]["activation"].as_str().unwrap(), "identity");
}

#[test]
fn perceptron_dims_too_few() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [36],
            "inputs": [[1.0]],
            "targets": [[1.0]],
            "epochs": 1
        }),
        serde_json::json!(7303),
    );

    let err = resp.error.expect("single dim should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("at least 2 dimensions"));
}

#[test]
fn perceptron_explicit_still_works() {
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [
                {"weights": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], "biases": [0.0, 0.0], "activation": "relu"}
            ],
            "inputs": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "targets": [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
            "epochs": 50,
            "learning_rate": 0.01
        }),
        serde_json::json!(7304),
    );

    assert!(resp.error.is_none(), "explicit layers still work: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert!(result["mse"].as_f64().unwrap().is_finite());
}

#[test]
fn perceptron_batch_256_performance() {
    let mut inputs: Vec<Vec<f64>> = Vec::new();
    let mut targets: Vec<Vec<f64>> = Vec::new();

    for i in 0..256 {
        let mut feature_vec = vec![0.0; 36];
        let domain = i % 16;
        feature_vec[domain] = 1.0;
        feature_vec[32] = (i as f64) / 256.0;
        feature_vec[33] = ((i * 7) % 100) as f64 / 100.0;
        feature_vec[34] = ((i * 3) % 50) as f64 / 50.0;
        feature_vec[35] = ((i * 11) % 100) as f64 / 100.0;
        inputs.push(feature_vec);

        let mut target = vec![0.0; 16];
        target[domain] = 1.0;
        targets.push(target);
    }

    let start = std::time::Instant::now();
    let resp = ml_mlp_train(
        &serde_json::json!({
            "layers": [36, 16],
            "inputs": inputs,
            "targets": targets,
            "learning_rate": 0.01,
            "epochs": 10
        }),
        serde_json::json!(7305),
    );
    let elapsed = start.elapsed();

    assert!(resp.error.is_none(), "batch-256 failed: {:?}", resp.error);
    assert!(
        elapsed.as_secs() < 1,
        "Training 256 samples × 10 epochs took {:?} (target <1s)",
        elapsed
    );

    let result = resp.result.unwrap();
    let mse = result["mse"].as_f64().unwrap();
    assert!(mse < 0.5, "MSE should decrease significantly with 256 samples, got {mse}");
}
