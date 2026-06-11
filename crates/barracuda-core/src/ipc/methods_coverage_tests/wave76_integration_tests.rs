// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wave 76 integration tests: remote mesh dispatch of ml.mlp_infer.
//!
//! Simulates the biomeOS v4.05 PerceptronAdvisor calling barraCuda via
//! Songbird capability routing. Validates the full cycle:
//! train → save → remote infer (with BTSP auth context).

use crate::ipc::methods::dispatch;
use serde_json::json;

use super::test_primal;

/// Simulates a realistic dispatch telemetry payload as biomeOS would generate.
fn mock_dispatch_telemetry(count: usize) -> Vec<serde_json::Value> {
    let methods = [
        "crypto.hash",
        "compute.fft",
        "stats.mean",
        "ml.forward",
        "tensor.matmul",
        "linalg.solve",
        "signal.bandpass",
        "graph.bfs",
    ];
    let providers = ["beardog", "songbird", "toadstool", "coralreef"];
    let gates = ["eastGate", "strandGate", "ironGate"];

    (0..count as i32)
        .map(|i| {
            let idx = i as usize;
            json!({
                "method": methods[idx % methods.len()],
                "owner": providers[idx % providers.len()],
                "latency_ms": f64::from(i).mul_add(0.3, 0.5),
                "success": !idx.is_multiple_of(7),
                "gate": gates[idx % gates.len()],
                "param_size_bytes": 1024.0 * f64::from(i + 1),
                "gate_load": 0.1 + f64::from(i % 10) * 0.08,
                "topology_affinity": if idx.is_multiple_of(3) { 0.9 } else { 0.5 }
            })
        })
        .collect()
}

/// Full end-to-end: train perceptron, save model, load + infer from "remote gate".
#[tokio::test]
async fn remote_mesh_infer_full_cycle() {
    let primal = test_primal();
    let records = mock_dispatch_telemetry(64);

    let train_resp = dispatch(
        &primal,
        "ml.perceptron_train",
        &json!({
            "records": records,
            "epochs": 20,
            "learning_rate": 0.01,
            "_auth": {"bearer": "btsp-eastgate-session-abc123"}
        }),
        json!(100),
    )
    .await;

    assert!(
        train_resp.error.is_none(),
        "train failed: {:?}",
        train_resp.error
    );
    let train_result = train_resp.result.unwrap();
    assert_eq!(train_result["records_processed"], 64);
    let providers = train_result["providers"].as_array().unwrap();
    assert_eq!(providers.len(), 4);
    assert!(train_result["mse"].as_f64().unwrap().is_finite());

    let tmp = std::env::temp_dir().join("barracuda_w76_integration_model.json");
    let tmp_str = tmp.to_string_lossy().to_string();

    let save_resp = dispatch(
        &primal,
        "ml.mlp_save",
        &json!({
            "model": {"layers": train_result["layers"]},
            "path": tmp_str,
            "_auth": {"bearer": "btsp-eastgate-session-abc123"}
        }),
        json!(101),
    )
    .await;

    assert!(
        save_resp.error.is_none(),
        "save failed: {:?}",
        save_resp.error
    );

    let infer_records = mock_dispatch_telemetry(8);
    let infer_resp = dispatch(
        &primal,
        "ml.mlp_infer",
        &json!({
            "model_path": tmp_str,
            "records": infer_records,
            "providers": ["beardog", "songbird", "toadstool", "coralreef"],
            "_auth": {"bearer": "btsp-eastgate-session-abc123"}
        }),
        json!(102),
    )
    .await;

    assert!(
        infer_resp.error.is_none(),
        "infer failed: {:?}",
        infer_resp.error
    );
    let infer_result = infer_resp.result.unwrap();
    assert_eq!(infer_result["records_processed"], 8);

    let results = infer_result["results"].as_array().unwrap();
    assert_eq!(results.len(), 8);

    for (i, entry) in results.iter().enumerate() {
        let scores = entry["scores"].as_array().unwrap();
        assert_eq!(scores.len(), 4, "record {i}: expected 4 provider scores");
        assert!(entry["best_index"].is_number());
        let provider = entry["best_provider"].as_str().unwrap();
        assert!(
            ["beardog", "songbird", "toadstool", "coralreef"].contains(&provider),
            "record {i}: unexpected provider {provider}"
        );
    }

    std::fs::remove_file(&tmp).ok();
}

/// biomeOS PerceptronAdvisor wire format: inline model with domain_index.
#[tokio::test]
async fn biomeos_perceptron_advisor_wire_format() {
    let primal = test_primal();
    let records = mock_dispatch_telemetry(32);

    let train_resp = dispatch(
        &primal,
        "ml.perceptron_train",
        &json!({
            "records": records,
            "epochs": 10,
            "_auth": {"bearer": "btsp-biomeos-l5-session"}
        }),
        json!(200),
    )
    .await;
    let train_result = train_resp.result.unwrap();

    let domain_index = json!({
        "crypto": 0,
        "compute": 1,
        "stats": 2,
        "ml": 3,
        "tensor": 4,
        "linalg": 5,
        "signal": 6,
        "graph": 7
    });

    let new_telemetry = mock_dispatch_telemetry(4);
    let infer_resp = dispatch(
        &primal,
        "ml.mlp_infer",
        &json!({
            "model": {"layers": train_result["layers"]},
            "records": new_telemetry,
            "providers": train_result["providers"],
            "domain_index": domain_index,
            "_auth": {"bearer": "btsp-biomeos-l5-session"}
        }),
        json!(201),
    )
    .await;

    assert!(
        infer_resp.error.is_none(),
        "biomeOS infer failed: {:?}",
        infer_resp.error
    );
    let result = infer_resp.result.unwrap();
    assert_eq!(result["records_processed"], 4);

    let results = result["results"].as_array().unwrap();
    for entry in results {
        assert!(entry["scores"].is_array());
        assert!(entry["best_index"].is_number());
        assert!(entry["best_provider"].is_string());
    }
}

/// Enforced mode: ml.mlp_infer requires bearer token (Dark Forest Invariant 3).
#[tokio::test]
async fn ml_infer_rejected_without_auth_in_enforced_mode() {
    use crate::ipc::method_gate::{CallerContext, ConnectionOrigin, EnforcementMode, MethodGate};

    let gate = MethodGate::new(EnforcementMode::Enforced);
    let caller = CallerContext {
        bearer_token: None,
        peer: None,
        origin: ConnectionOrigin::Remote,
    };
    let id = json!(300);
    let result = gate.check("ml.mlp_infer", &caller, &id);
    assert!(
        result.is_err(),
        "ml.mlp_infer should be rejected without auth"
    );
}

/// Mesh health probe is public — accessible without bearer token.
#[tokio::test]
async fn mesh_health_public_no_auth_needed() {
    use crate::ipc::method_gate::{CallerContext, ConnectionOrigin, EnforcementMode, MethodGate};

    let gate = MethodGate::new(EnforcementMode::Enforced);
    let caller = CallerContext {
        bearer_token: None,
        peer: None,
        origin: ConnectionOrigin::Remote,
    };
    let id = json!(301);
    assert!(gate.check("mesh.health", &caller, &id).is_ok());
    assert!(gate.check("mesh.trust_verify", &caller, &id).is_ok());
}

/// Save as bincode, load with auto-detection, verify roundtrip.
#[tokio::test]
async fn binary_format_save_load_roundtrip() {
    let primal = test_primal();
    let records = mock_dispatch_telemetry(32);

    let train_resp = dispatch(
        &primal,
        "ml.perceptron_train",
        &json!({
            "records": records,
            "epochs": 10,
            "_auth": {"bearer": "tok"}
        }),
        json!(500),
    )
    .await;
    let trained = train_resp.result.unwrap();

    let tmp = std::env::temp_dir().join("barracuda_w76_bincode_model.bin");
    let tmp_str = tmp.to_string_lossy().to_string();

    let save_resp = dispatch(
        &primal,
        "ml.mlp_save",
        &json!({
            "model": {"layers": trained["layers"]},
            "path": tmp_str,
            "format": "bincode",
            "_auth": {"bearer": "tok"}
        }),
        json!(501),
    )
    .await;

    assert!(
        save_resp.error.is_none(),
        "bincode save failed: {:?}",
        save_resp.error
    );
    let save_result = save_resp.result.unwrap();
    assert_eq!(save_result["format"], "bincode");
    let json_size_estimate = serde_json::to_string(&trained["layers"]).unwrap().len();
    let bin_size = save_result["bytes_written"].as_u64().unwrap() as usize;
    assert!(
        bin_size < json_size_estimate,
        "bincode ({bin_size}) should be smaller than JSON (~{json_size_estimate})"
    );

    let load_resp = dispatch(
        &primal,
        "ml.mlp_load",
        &json!({
            "path": tmp_str,
            "_auth": {"bearer": "tok"}
        }),
        json!(502),
    )
    .await;

    assert!(
        load_resp.error.is_none(),
        "bincode load failed: {:?}",
        load_resp.error
    );
    let load_result = load_resp.result.unwrap();
    assert_eq!(load_result["format"], "bincode");
    assert_eq!(
        load_result["layer_count"],
        trained["layers"].as_array().unwrap().len()
    );

    let infer_telemetry = mock_dispatch_telemetry(4);
    let infer_resp = dispatch(
        &primal,
        "ml.mlp_infer",
        &json!({
            "model_path": tmp_str,
            "records": infer_telemetry,
            "providers": trained["providers"],
            "_auth": {"bearer": "tok"}
        }),
        json!(503),
    )
    .await;

    assert!(
        infer_resp.error.is_none(),
        "infer from bincode model failed: {:?}",
        infer_resp.error
    );
    assert_eq!(infer_resp.result.unwrap()["records_processed"], 4);

    std::fs::remove_file(&tmp).ok();
}

/// Verify ml.mlp_infer handles varied batch sizes correctly.
#[tokio::test]
async fn infer_batch_scaling() {
    let primal = test_primal();

    let train_records = mock_dispatch_telemetry(100);
    let train_resp = dispatch(
        &primal,
        "ml.perceptron_train",
        &json!({
            "records": train_records,
            "epochs": 5,
            "_auth": {"bearer": "tok"}
        }),
        json!(400),
    )
    .await;
    let trained = train_resp.result.unwrap();

    for batch_size in [1, 4, 16, 64, 256] {
        let batch = mock_dispatch_telemetry(batch_size);
        let resp = dispatch(
            &primal,
            "ml.mlp_infer",
            &json!({
                "model": {"layers": trained["layers"]},
                "records": batch,
                "providers": trained["providers"],
                "_auth": {"bearer": "tok"}
            }),
            json!(401 + batch_size as i64),
        )
        .await;

        assert!(
            resp.error.is_none(),
            "batch {batch_size} failed: {:?}",
            resp.error
        );
        let result = resp.result.unwrap();
        assert_eq!(
            result["records_processed"].as_u64().unwrap(),
            batch_size as u64
        );
    }
}
