// SPDX-License-Identifier: AGPL-3.0-or-later
//! Model persistence: save and load trained models to/from disk.

use super::super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use barracuda::nn::simple_mlp::SimpleMlp;
use serde_json::Value;
use std::path::Path;

/// `ml.mlp_save` — Persist a trained model to disk.
///
/// Serializes the model and writes to the specified path.
/// Supports `"format": "json"` (default, human-readable) or
/// `"format": "bincode"` (compact binary with BLAKE3 integrity header).
///
/// Wire contract:
/// ```json
/// { "model": {"layers": [...]},
///   "path": "/data/gate/neural_routing_perceptron.bin",
///   "format": "bincode" }
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
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Path traversal (..) not permitted");
    }

    let format = params
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("json");

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

    let (data, written_format) = match format {
        "bincode" | "binary" => match mlp.to_binary() {
            Ok(bytes) => (bytes, "bincode"),
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("Binary serialization failed: {e}"),
                );
            }
        },
        _ => match mlp.to_json() {
            Ok(s) => (s.into_bytes(), "json"),
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("JSON serialization failed: {e}"),
                );
            }
        },
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

    match std::fs::write(path, &data) {
        Ok(()) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "path": path_str,
                "bytes_written": data.len(),
                "format": written_format,
            }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Write failed: {e}")),
    }
}

/// `ml.mlp_load` — Load a persisted model from disk.
///
/// Auto-detects format from file content: if the file starts with `BCML`
/// magic bytes, loads as bincode with BLAKE3 verification. Otherwise
/// falls back to JSON parsing. Returns the full model structure.
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
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Path traversal (..) not permitted");
    }

    let path = Path::new(path_str);
    if !path.exists() {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Model file not found: {path_str}"),
        );
    }

    let data = match std::fs::read(path) {
        Ok(d) => d,
        Err(e) => {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Read failed: {e}"));
        }
    };

    let mlp = match SimpleMlp::from_auto(&data) {
        Ok(m) => m,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Invalid model format: {e}"),
            );
        }
    };

    let detected_format = if data.len() >= 4 && &data[0..4] == b"BCML" {
        "bincode"
    } else {
        "json"
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
            "format": detected_format,
        }),
    )
}
