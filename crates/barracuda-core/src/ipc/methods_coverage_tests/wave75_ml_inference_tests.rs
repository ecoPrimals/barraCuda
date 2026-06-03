// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wave 75 tests: ml.mlp_infer, ml.mlp_save, ml.mlp_load.

use crate::ipc::methods::dispatch;
use serde_json::json;

use super::test_primal;

fn trained_model_json() -> serde_json::Value {
    let weights: Vec<Vec<f64>> = (0..2)
        .map(|row| (0..36).map(|col| 0.01 * (row * 36 + col) as f64).collect())
        .collect();
    json!({
        "layers": [{
            "weight": weights,
            "bias": [0.01, 0.02],
            "activation": "sigmoid"
        }]
    })
}

#[tokio::test]
async fn mlp_infer_inline_model() {
    let primal = test_primal();
    let params = json!({
        "model": trained_model_json(),
        "records": [
            {"method": "stats.mean", "latency_ms": 1.0, "success": true},
            {"method": "ml.train", "latency_ms": 2.0, "success": false}
        ]
    });
    let resp = dispatch(&primal, "ml.mlp_infer", &params, json!(1)).await;
    let result = resp.result.unwrap();
    assert_eq!(result["records_processed"], 2);
    let results = result["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert!(results[0]["scores"].is_array());
    assert!(results[0]["best_index"].is_number());
}

#[tokio::test]
async fn mlp_infer_with_providers() {
    let primal = test_primal();
    let params = json!({
        "model": trained_model_json(),
        "records": [{"method": "stats.mean", "latency_ms": 1.0, "success": true}],
        "providers": ["beardog", "songbird"]
    });
    let resp = dispatch(&primal, "ml.mlp_infer", &params, json!(2)).await;
    let result = resp.result.unwrap();
    let first = &result["results"][0];
    assert!(first.get("best_provider").is_some());
    let provider = first["best_provider"].as_str().unwrap();
    assert!(provider == "beardog" || provider == "songbird");
}

#[tokio::test]
async fn mlp_infer_missing_model() {
    let primal = test_primal();
    let params = json!({
        "records": [{"method": "stats.mean", "latency_ms": 1.0, "success": true}]
    });
    let resp = dispatch(&primal, "ml.mlp_infer", &params, json!(3)).await;
    assert!(resp.error.is_some());
    assert!(resp.error.unwrap().message.contains("model"));
}

#[tokio::test]
async fn mlp_infer_missing_records() {
    let primal = test_primal();
    let params = json!({"model": trained_model_json()});
    let resp = dispatch(&primal, "ml.mlp_infer", &params, json!(4)).await;
    assert!(resp.error.is_some());
    assert!(resp.error.unwrap().message.contains("records"));
}

#[tokio::test]
async fn mlp_infer_empty_records() {
    let primal = test_primal();
    let params = json!({"model": trained_model_json(), "records": []});
    let resp = dispatch(&primal, "ml.mlp_infer", &params, json!(5)).await;
    assert!(resp.error.is_some());
    assert!(resp.error.unwrap().message.contains("non-empty"));
}

#[tokio::test]
async fn mlp_save_and_load_roundtrip() {
    let primal = test_primal();
    let tmp = std::env::temp_dir().join("barracuda_test_wave75_save.json");
    let tmp_str = tmp.to_string_lossy().to_string();

    let save_params = json!({
        "model": trained_model_json(),
        "path": tmp_str,
    });
    let save_resp = dispatch(&primal, "ml.mlp_save", &save_params, json!(6)).await;
    let save_result = save_resp.result.unwrap();
    assert_eq!(save_result["path"], tmp_str);
    assert!(save_result["bytes_written"].as_u64().unwrap() > 0);

    let load_params = json!({"path": tmp_str});
    let load_resp = dispatch(&primal, "ml.mlp_load", &load_params, json!(7)).await;
    let load_result = load_resp.result.unwrap();
    assert_eq!(load_result["layer_count"], 1);
    assert!(load_result["layers"].is_array());

    std::fs::remove_file(&tmp).ok();
}

#[tokio::test]
async fn mlp_save_path_traversal_rejected() {
    let primal = test_primal();
    let params = json!({
        "model": trained_model_json(),
        "path": "/tmp/../etc/evil.json",
    });
    let resp = dispatch(&primal, "ml.mlp_save", &params, json!(8)).await;
    assert!(resp.error.is_some());
    assert!(resp.error.unwrap().message.contains("traversal"));
}

#[tokio::test]
async fn mlp_load_missing_file() {
    let primal = test_primal();
    let params = json!({"path": "/nonexistent/path/model.json"});
    let resp = dispatch(&primal, "ml.mlp_load", &params, json!(9)).await;
    assert!(resp.error.is_some());
    assert!(resp.error.unwrap().message.contains("not found"));
}

#[tokio::test]
async fn mlp_load_path_traversal_rejected() {
    let primal = test_primal();
    let params = json!({"path": "/tmp/../../etc/passwd"});
    let resp = dispatch(&primal, "ml.mlp_load", &params, json!(10)).await;
    assert!(resp.error.is_some());
    assert!(resp.error.unwrap().message.contains("traversal"));
}

#[tokio::test]
async fn mlp_infer_from_saved_model() {
    let primal = test_primal();
    let tmp = std::env::temp_dir().join("barracuda_test_wave75_infer.json");
    let tmp_str = tmp.to_string_lossy().to_string();

    let save_params = json!({
        "model": trained_model_json(),
        "path": tmp_str,
    });
    dispatch(&primal, "ml.mlp_save", &save_params, json!(11)).await;

    let infer_params = json!({
        "model_path": tmp_str,
        "records": [{"method": "stats.mean", "latency_ms": 0.5, "success": true}],
    });
    let resp = dispatch(&primal, "ml.mlp_infer", &infer_params, json!(12)).await;
    let result = resp.result.unwrap();
    assert_eq!(result["records_processed"], 1);

    std::fs::remove_file(&tmp).ok();
}
