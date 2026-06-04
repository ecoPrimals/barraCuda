// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model persistence: save and load trained models to/from disk.

use super::super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use barracuda::nn::simple_mlp::SimpleMlp;
use serde_json::Value;
use std::path::Path;

/// `ml.mlp_save` — Persist a trained model to disk.
///
/// Serializes the model as JSON and writes to the specified path.
/// The path must be within allowed directories (no traversal).
///
/// Wire contract:
/// ```json
/// { "model": {"layers": [...]}, "path": "/data/gate/neural_routing_perceptron.bin" }
/// ```
pub(in crate::ipc::methods) fn ml_mlp_save(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(model_val) = params.get("model") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: model");
    };
    let Some(path_str) = params.get("path").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: path");
    };

    let path = Path::new(path_str);
    if path_str.contains("..") {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Path traversal (..) not permitted",
        );
    }

    let mlp: SimpleMlp = match serde_json::from_value(model_val.clone()) {
        Ok(m) => m,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Invalid model structure: {e}"),
            );
        }
    };

    let json_str = match mlp.to_json() {
        Ok(s) => s,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Serialization failed: {e}"),
            );
        }
    };

    if let Some(parent) = path.parent() {
        if !parent.exists() {
            if let Err(e) = std::fs::create_dir_all(parent) {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("Cannot create directory {}: {e}", parent.display()),
                );
            }
        }
    }

    match std::fs::write(path, &json_str) {
        Ok(()) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "path": path_str,
                "bytes_written": json_str.len(),
                "format": "json",
            }),
        ),
        Err(e) => JsonRpcResponse::error(
            id,
            INTERNAL_ERROR,
            format!("Write failed: {e}"),
        ),
    }
}

/// `ml.mlp_load` — Load a persisted model from disk.
///
/// Reads and deserializes a model from the specified path.
/// Returns the full model structure (layers with weights/biases/activation).
///
/// Wire contract:
/// ```json
/// { "path": "/data/gate/neural_routing_perceptron.bin" }
/// ```
pub(in crate::ipc::methods) fn ml_mlp_load(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(path_str) = params.get("path").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: path");
    };

    if path_str.contains("..") {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Path traversal (..) not permitted",
        );
    }

    let path = Path::new(path_str);
    if !path.exists() {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Model file not found: {path_str}"),
        );
    }

    let json_str = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Read failed: {e}"),
            );
        }
    };

    let mlp: SimpleMlp = match SimpleMlp::from_json(&json_str) {
        Ok(m) => m,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Invalid model format: {e}"),
            );
        }
    };

    let layers: Vec<Value> = mlp
        .layers
        .iter()
        .map(|l| {
            serde_json::json!({
                "weights": l.weight,
                "biases": l.bias,
                "activation": format!("{:?}", l.activation).to_lowercase(),
            })
        })
        .collect();

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "layers": layers,
            "path": path_str,
            "layer_count": layers.len(),
        }),
    )
}
