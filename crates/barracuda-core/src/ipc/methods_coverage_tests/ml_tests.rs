// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for ml.mlp_forward and ml.attention handlers (Sprint 45).

use crate::ipc::jsonrpc;

use super::super::ml::{ml_attention, ml_mlp_forward};

// ── ml.mlp_forward ─────────────────────────────────────────────────

#[test]
fn ml_mlp_forward_missing_input() {
    let resp = ml_mlp_forward(&serde_json::json!({"layers": []}), serde_json::json!(300));
    let err = resp.error.expect("missing input");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_mlp_forward_missing_layers() {
    let resp = ml_mlp_forward(
        &serde_json::json!({"input": [1.0, 2.0]}),
        serde_json::json!(301),
    );
    let err = resp.error.expect("missing layers");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_mlp_forward_empty_layers() {
    let resp = ml_mlp_forward(
        &serde_json::json!({"input": [1.0], "layers": []}),
        serde_json::json!(302),
    );
    let err = resp.error.expect("empty layers");
    assert!(err.message.contains("non-empty"));
}

#[test]
fn ml_mlp_forward_unknown_activation() {
    let resp = ml_mlp_forward(
        &serde_json::json!({
            "input": [1.0],
            "layers": [{"weights": [[1.0]], "biases": [0.0], "activation": "swish"}]
        }),
        serde_json::json!(303),
    );
    let err = resp.error.expect("unknown activation");
    assert!(err.message.contains("swish"));
}

#[test]
fn ml_mlp_forward_happy_path() {
    let resp = ml_mlp_forward(
        &serde_json::json!({
            "input": [1.0, 2.0],
            "layers": [
                {"weights": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "biases": [0.0, 0.0, -0.5], "activation": "relu"},
                {"weights": [[1.0, 1.0, 1.0]], "biases": [0.0], "activation": "identity"}
            ]
        }),
        serde_json::json!(304),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(arr.len(), 1);
    assert!((arr[0].as_f64().unwrap() - 5.5).abs() < 1e-10);
}

#[test]
fn ml_mlp_forward_shape_mismatch() {
    let resp = ml_mlp_forward(
        &serde_json::json!({
            "input": [1.0],
            "layers": [{"weights": [[1.0, 2.0]], "biases": [0.0, 0.0]}]
        }),
        serde_json::json!(305),
    );
    let err = resp.error.expect("weights/biases mismatch");
    assert!(err.message.contains("weights rows"));
}

// ── ml.attention ───────────────────────────────────────────────────

#[test]
fn ml_attention_missing_q() {
    let resp = ml_attention(
        &serde_json::json!({"k": [[1.0]], "v": [[1.0]]}),
        serde_json::json!(310),
    );
    let err = resp.error.expect("missing q");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_attention_missing_k() {
    let resp = ml_attention(
        &serde_json::json!({"q": [[1.0]], "v": [[1.0]]}),
        serde_json::json!(311),
    );
    let err = resp.error.expect("missing k");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_attention_missing_v() {
    let resp = ml_attention(
        &serde_json::json!({"q": [[1.0]], "k": [[1.0]]}),
        serde_json::json!(312),
    );
    let err = resp.error.expect("missing v");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn ml_attention_dim_mismatch() {
    let resp = ml_attention(
        &serde_json::json!({
            "q": [[1.0, 2.0]],
            "k": [[1.0]],
            "v": [[1.0]]
        }),
        serde_json::json!(313),
    );
    let err = resp.error.expect("dimension mismatch");
    assert!(err.message.contains("inner dimension"));
}

#[test]
fn ml_attention_happy_path() {
    let resp = ml_attention(
        &serde_json::json!({
            "q": [[1.0, 0.0], [0.0, 1.0]],
            "k": [[1.0, 0.0], [0.0, 1.0]],
            "v": [[10.0, 20.0], [30.0, 40.0]]
        }),
        serde_json::json!(314),
    );
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let result = r["result"].as_array().unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(r["seq_q"].as_u64().unwrap(), 2);
    assert_eq!(r["d_k"].as_u64().unwrap(), 2);
    assert_eq!(r["d_v"].as_u64().unwrap(), 2);
}

#[test]
fn ml_attention_single_token() {
    let resp = ml_attention(
        &serde_json::json!({
            "q": [[1.0, 0.0]],
            "k": [[1.0, 0.0]],
            "v": [[42.0, 7.0]]
        }),
        serde_json::json!(315),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert_eq!(result.len(), 1);
    let row = result[0].as_array().unwrap();
    assert!((row[0].as_f64().unwrap() - 42.0).abs() < 1e-8);
    assert!((row[1].as_f64().unwrap() - 7.0).abs() < 1e-8);
}
