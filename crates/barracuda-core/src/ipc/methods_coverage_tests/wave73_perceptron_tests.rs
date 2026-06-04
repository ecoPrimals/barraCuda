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

use super::super::ml::{ml_mlp_train, ml_perceptron_train};

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

// ── ml.perceptron_train pipeline tests ──────────────────────────────

#[test]
fn perceptron_pipeline_from_telemetry() {
    let records: Vec<serde_json::Value> = (0..64_i32)
        .map(|i| {
            let domains = ["crypto", "compute", "storage", "network"];
            let providers = ["beardog", "songbird", "toadstool", "coralreef"];
            let idx = i as usize;
            serde_json::json!({
                "method": format!("{}.op{}", domains[idx % 4], i),
                "owner": providers[idx % 4],
                "latency_ms": f64::from(i).mul_add(0.5, 0.1),
                "success": !idx.is_multiple_of(5),
                "gate": "eastGate"
            })
        })
        .collect();

    let resp = ml_perceptron_train(
        &serde_json::json!({
            "records": records,
            "learning_rate": 0.01,
            "epochs": 10
        }),
        serde_json::json!(7400),
    );

    assert!(resp.error.is_none(), "pipeline failed: {:?}", resp.error);
    let result = resp.result.unwrap();

    assert_eq!(result["records_processed"].as_u64().unwrap(), 64);
    assert_eq!(result["epochs"].as_u64().unwrap(), 10);
    assert!(result["mse"].as_f64().unwrap().is_finite());

    let providers = result["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 4);

    let domains = result["domains"].as_array().unwrap();
    assert_eq!(domains.len(), 4);

    let layers = result["layers"].as_array().unwrap();
    assert_eq!(layers.len(), 1);
    let weights = layers[0]["weights"].as_array().unwrap();
    assert_eq!(weights.len(), 4, "output dim = number of providers");
    assert_eq!(weights[0].as_array().unwrap().len(), 36, "input dim = 36");
}

#[test]
fn perceptron_pipeline_with_output_path() {
    let tmp = std::env::temp_dir().join("barracuda_perceptron_test.json");
    let records: Vec<serde_json::Value> = (0..16_i32)
        .map(|i| {
            serde_json::json!({
                "method": format!("crypto.hash{}", i),
                "owner": if i % 2 == 0 { "beardog" } else { "songbird" },
                "latency_ms": 1.0 + f64::from(i),
                "success": true,
                "gate": "strandGate"
            })
        })
        .collect();

    let resp = ml_perceptron_train(
        &serde_json::json!({
            "records": records,
            "epochs": 5,
            "output_path": tmp.to_str().unwrap()
        }),
        serde_json::json!(7401),
    );

    assert!(resp.error.is_none(), "pipeline with output failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert_eq!(
        result["output_path"].as_str().unwrap(),
        tmp.to_str().unwrap()
    );

    assert!(tmp.exists(), "weights file should be written");
    let contents = std::fs::read_to_string(&tmp).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();
    assert!(!parsed["layers"].as_array().unwrap().is_empty());

    std::fs::remove_file(&tmp).ok();
}

#[test]
fn perceptron_pipeline_empty_records() {
    let resp = ml_perceptron_train(
        &serde_json::json!({"records": []}),
        serde_json::json!(7402),
    );
    let err = resp.error.expect("empty records should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn perceptron_pipeline_missing_records() {
    let resp = ml_perceptron_train(
        &serde_json::json!({"epochs": 10}),
        serde_json::json!(7403),
    );
    let err = resp.error.expect("missing records should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn perceptron_pipeline_single_provider() {
    let records: Vec<serde_json::Value> = (0..10_i32)
        .map(|i| {
            serde_json::json!({
                "method": "crypto.hash",
                "owner": "beardog",
                "latency_ms": f64::from(i).mul_add(0.1, 0.5),
                "success": true,
                "gate": "strandGate"
            })
        })
        .collect();

    let resp = ml_perceptron_train(
        &serde_json::json!({"records": records, "epochs": 5}),
        serde_json::json!(7404),
    );

    assert!(resp.error.is_none(), "single provider failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    let providers = result["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].as_str().unwrap(), "beardog");
}

#[test]
fn perceptron_pipeline_reward_signal() {
    let records = vec![
        serde_json::json!({"method":"crypto.hash","owner":"beardog","latency_ms":0.1,"success":true,"gate":"eastGate"}),
        serde_json::json!({"method":"crypto.hash","owner":"songbird","latency_ms":100.0,"success":true,"gate":"eastGate"}),
        serde_json::json!({"method":"crypto.hash","owner":"beardog","latency_ms":0.2,"success":true,"gate":"eastGate"}),
        serde_json::json!({"method":"crypto.hash","owner":"songbird","latency_ms":50.0,"success":false,"gate":"eastGate"}),
    ];

    let resp = ml_perceptron_train(
        &serde_json::json!({"records": records, "epochs": 50, "learning_rate": 0.1}),
        serde_json::json!(7405),
    );

    assert!(resp.error.is_none(), "reward signal test failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    let mse = result["mse"].as_f64().unwrap();
    assert!(mse.is_finite());
}

#[test]
fn perceptron_pipeline_full_feature_fields() {
    let records = vec![
        serde_json::json!({
            "method": "crypto.hash",
            "owner": "beardog",
            "latency_ms": 0.5,
            "success": true,
            "gate": "strandGate",
            "param_size_bytes": 1024,
            "gate_load": 50,
            "topology_affinity": 0.95
        }),
        serde_json::json!({
            "method": "compute.dispatch",
            "owner": "songbird",
            "latency_ms": 15.0,
            "success": true,
            "gate": "eastGate",
            "param_size_bytes": 4096000,
            "gate_load": 800,
            "topology_affinity": 0.3
        }),
        serde_json::json!({
            "method": "storage.put",
            "owner": "beardog",
            "latency_ms": 2.0,
            "success": false,
            "gate": "strandGate",
            "param_size_bytes": 512,
            "gate_load": 100,
            "topology_affinity": 0.0
        }),
        serde_json::json!({
            "method": "crypto.verify",
            "owner": "songbird",
            "latency_ms": 0.1,
            "success": true,
            "gate": "eastGate",
            "param_size_bytes": 256,
            "gate_load": 20,
            "topology_affinity": 0.85
        }),
    ];

    let resp = ml_perceptron_train(
        &serde_json::json!({"records": records, "epochs": 20, "learning_rate": 0.05}),
        serde_json::json!(7406),
    );

    assert!(resp.error.is_none(), "full fields test failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert_eq!(result["records_processed"].as_u64().unwrap(), 4);
    assert!(result["mse"].as_f64().unwrap().is_finite());

    let providers = result["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 2);
    let domains = result["domains"].as_array().unwrap();
    assert_eq!(domains.len(), 3);
}
