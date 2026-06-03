// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML inference handlers for JSON-RPC IPC.
//!
//! Inline-data CPU paths for lightweight ML operations suitable for
//! composition graph nodes. GPU tensor ops live in `tensor.rs`.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::params::{extract_f64_array, extract_matrix};
use barracuda::nn::esn_classifier::EsnClassifier;
use barracuda::nn::simple_mlp::{Activation, DenseLayer, SimpleMlp, TrainConfig};
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

fn parse_activation(s: &str) -> Option<Activation> {
    match s {
        "relu" => Some(Activation::Relu),
        "tanh" => Some(Activation::Tanh),
        "sigmoid" => Some(Activation::Sigmoid),
        "gelu" => Some(Activation::Gelu),
        "identity" | "none" | "linear" => Some(Activation::Identity),
        _ => None,
    }
}

/// `ml.mlp_forward` — inline-data MLP forward pass (CPU).
///
/// Params: `input` (array), `layers` (array of `{weights, biases, activation}`).
/// Each layer: `weights` is 2D (out×in), `biases` is 1D (out), `activation` is a string.
pub(super) fn ml_mlp_forward(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(input) = extract_f64_array(params, "input") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: input (array)");
    };
    let Some(layers_val) = params.get("layers").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: layers (array of layer specs)",
        );
    };
    if layers_val.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "layers must be non-empty");
    }

    let mut dense_layers = Vec::with_capacity(layers_val.len());
    for (i, layer_val) in layers_val.iter().enumerate() {
        let Some(weights) = extract_matrix(layer_val, "weights") else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Layer {i}: missing weights (2D array)"),
            );
        };
        let Some(biases) = extract_f64_array(layer_val, "biases") else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Layer {i}: missing biases (array)"),
            );
        };
        if weights.len() != biases.len() {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!(
                    "Layer {i}: weights rows ({}) != biases length ({})",
                    weights.len(),
                    biases.len()
                ),
            );
        }
        let act_str = layer_val
            .get("activation")
            .and_then(|v| v.as_str())
            .unwrap_or("identity");
        let Some(activation) = parse_activation(act_str) else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Layer {i}: unknown activation \"{act_str}\""),
            );
        };
        dense_layers.push(DenseLayer {
            weight: weights,
            bias: biases,
            activation,
        });
    }

    let mlp = SimpleMlp::new(dense_layers);
    let output = mlp.forward(&input);
    JsonRpcResponse::success(id, serde_json::json!({ "result": output }))
}

/// `ml.attention` — inline-data scaled dot-product attention (CPU).
///
/// Params: `q` (2D), `k` (2D), `v` (2D). Computes softmax(QK^T / √d_k) · V.
pub(super) fn ml_attention(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(q) = extract_matrix(params, "q") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: q (2D array)");
    };
    let Some(k) = extract_matrix(params, "k") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: k (2D array)");
    };
    let Some(v) = extract_matrix(params, "v") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: v (2D array)");
    };

    if q.is_empty() || k.is_empty() || v.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "q, k, v must be non-empty");
    }

    let d_k = q[0].len();
    if d_k == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Key dimension must be > 0");
    }
    let seq_q = q.len();
    let seq_k = k.len();

    if k.iter().any(|row| row.len() != d_k) || q.iter().any(|row| row.len() != d_k) {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "q and k must have matching inner dimension",
        );
    }
    let d_v = v[0].len();
    if v.len() != seq_k || v.iter().any(|row| row.len() != d_v) {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "v must have same sequence length as k with consistent column count",
        );
    }

    let scale = 1.0 / (d_k as f64).sqrt();

    // QK^T / √d_k  →  (seq_q × seq_k)
    let mut scores: Vec<Vec<f64>> = Vec::with_capacity(seq_q);
    for qi in &q {
        let row: Vec<f64> = k
            .iter()
            .map(|ki| qi.iter().zip(ki).map(|(a, b)| a * b).sum::<f64>() * scale)
            .collect();
        scores.push(row);
    }

    // Row-wise softmax
    for row in &mut scores {
        let max_val = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }

    // Scores · V  →  (seq_q × d_v)
    let mut output: Vec<Vec<f64>> = Vec::with_capacity(seq_q);
    for row in &scores {
        let mut out_row = vec![0.0; d_v];
        for (j, &w) in row.iter().enumerate() {
            for (c, val) in v[j].iter().enumerate() {
                out_row[c] = w.mul_add(*val, out_row[c]);
            }
        }
        output.push(out_row);
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "result": output,
            "seq_q": seq_q,
            "seq_k": seq_k,
            "d_k": d_k,
            "d_v": d_v,
        }),
    )
}

/// `ml.esn_predict` — stateless ESN prediction (Path A, client-managed state).
///
/// Accepts serialized ESN weights (JSON string from `EsnClassifier::to_json`),
/// an optional reservoir state snapshot, and the current input vector.
/// Returns the prediction and the new reservoir state for the next call.
///
/// Params:
/// - `weights_json` (string): serialized ESN weights from `to_json()`
/// - `input` (array): input vector of length `input_size`
/// - `state` (array, optional): reservoir state from a previous call (zeros if omitted)
pub(super) fn ml_esn_predict(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(weights_json) = params.get("weights_json").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: weights_json (string from EsnClassifier::to_json())",
        );
    };
    let Some(input) = extract_f64_array(params, "input") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: input (array)");
    };

    let mut esn = match EsnClassifier::from_json(weights_json) {
        Ok(e) => e,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Failed to parse weights_json: {e}"),
            );
        }
    };

    if let Some(state_arr) = extract_f64_array(params, "state") {
        if state_arr.len() != esn.config.reservoir_size {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!(
                    "state length ({}) != reservoir_size ({})",
                    state_arr.len(),
                    esn.config.reservoir_size
                ),
            );
        }
        esn.set_state(&state_arr);
    }

    match esn.predict(&input) {
        Ok(prediction) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "prediction": prediction,
                "state": esn.get_state(),
            }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("ESN predict failed: {e}")),
    }
}

/// `ml.mlp_train` — inline-data MLP training via backpropagation (CPU).
///
/// Params:
/// - `layers` (array of `{weights, biases, activation}`): initial network architecture
/// - `inputs` (array of arrays): training input vectors
/// - `targets` (array of arrays): target output vectors
/// - `learning_rate` (number, optional, default 0.01)
/// - `epochs` (integer, optional, default 100)
///
/// Returns trained weights (same format as `ml.mlp_forward` layers) and final MSE.
/// Train an MLP via SGD backpropagation.
///
/// Supports two forms for the `layers` parameter:
/// - **Shorthand (dims)**: `[36, 16]` — creates a network with Xavier-init random weights.
///   Optional top-level `"activation"` (default `"sigmoid"`) sets hidden activations.
/// - **Explicit (weights)**: `[{"weights":..., "biases":..., "activation":...}]` — resumes
///   training from existing weights (original contract).
pub(super) fn ml_mlp_train(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(layers_val) = params.get("layers").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: layers (array of dims or layer specs)",
        );
    };
    if layers_val.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "layers must be non-empty");
    }

    let is_dims_shorthand = layers_val[0].is_number();

    let mlp = if is_dims_shorthand {
        let dims: Vec<usize> = layers_val
            .iter()
            .filter_map(|v| v.as_u64().map(|n| n as usize))
            .collect();
        if dims.len() < 2 {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                "Shorthand layers requires at least 2 dimensions [input_dim, output_dim]",
            );
        }
        let act_str = params
            .get("activation")
            .and_then(|v| v.as_str())
            .unwrap_or("sigmoid");
        let Some(activation) = parse_activation(act_str) else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Unknown activation \"{act_str}\""),
            );
        };
        SimpleMlp::from_dims(&dims, activation)
    } else {
        let mut dense_layers = Vec::with_capacity(layers_val.len());
        for (i, layer_val) in layers_val.iter().enumerate() {
            let Some(weights) = extract_matrix(layer_val, "weights") else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Layer {i}: missing weights (2D array)"),
                );
            };
            let Some(biases) = extract_f64_array(layer_val, "biases") else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Layer {i}: missing biases (array)"),
                );
            };
            if weights.len() != biases.len() {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!(
                        "Layer {i}: weights rows ({}) != biases length ({})",
                        weights.len(),
                        biases.len()
                    ),
                );
            }
            let act_str = layer_val
                .get("activation")
                .and_then(|v| v.as_str())
                .unwrap_or("identity");
            let Some(activation) = parse_activation(act_str) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Layer {i}: unknown activation \"{act_str}\""),
                );
            };
            dense_layers.push(DenseLayer {
                weight: weights,
                bias: biases,
                activation,
            });
        }
        SimpleMlp::new(dense_layers)
    };

    let Some(inputs_val) = params.get("inputs").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: inputs (array of arrays)",
        );
    };
    let Some(targets_val) = params.get("targets").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: targets (array of arrays)",
        );
    };

    let inputs: Vec<Vec<f64>> = inputs_val
        .iter()
        .filter_map(|v| {
            v.as_array()
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
        })
        .collect();
    let targets: Vec<Vec<f64>> = targets_val
        .iter()
        .filter_map(|v| {
            v.as_array()
                .map(|a| a.iter().filter_map(|x| x.as_f64()).collect())
        })
        .collect();

    let learning_rate = params
        .get("learning_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01);
    #[expect(clippy::cast_possible_truncation, reason = "epochs is a count")]
    let epochs = params.get("epochs").and_then(|v| v.as_u64()).unwrap_or(100) as usize;

    let config = TrainConfig {
        learning_rate,
        epochs,
    };

    let mut mlp = mlp;
    match mlp.train(&inputs, &targets, &config) {
        Ok(mse) => {
            let trained_layers: Vec<Value> = mlp
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
                    "layers": trained_layers,
                    "mse": mse,
                    "epochs": epochs,
                }),
            )
        }
        Err(e) => JsonRpcResponse::error(id, INVALID_PARAMS, format!("Training failed: {e}")),
    }
}

/// End-to-end perceptron training pipeline for biomeOS L5 Neural API routing.
///
/// Accepts raw dispatch telemetry records, extracts 36-dim feature vectors per
/// `NEURAL_API_PERCEPTRON_DESIGN.md`, trains a single-layer perceptron, and
/// optionally serializes weights to disk for biomeOS consumption.
///
/// Wire contract:
/// ```json
/// { "records": [{"method":"crypto.hash","owner":"beardog","latency_ms":0.8,"success":true,"gate":"eastGate"},...],
///   "learning_rate": 0.01, "epochs": 10,
///   "output_path": "/path/to/neural_routing_perceptron.bin" }
/// ```
pub(super) fn ml_perceptron_train(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(records) = params.get("records").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: records (array of dispatch telemetry objects)",
        );
    };
    if records.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "records must be non-empty");
    }

    let learning_rate = params
        .get("learning_rate")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.01);
    #[expect(clippy::cast_possible_truncation, reason = "epochs is a count")]
    let epochs = params
        .get("epochs")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;
    let output_path = params.get("output_path").and_then(|v| v.as_str());

    let mut domain_index: HashMap<String, usize> = HashMap::new();
    let mut provider_index: HashMap<String, usize> = HashMap::new();

    for record in records {
        if let Some(method) = record.get("method").and_then(|v| v.as_str()) {
            let domain = method.split('.').next().unwrap_or(method);
            let len = domain_index.len();
            domain_index.entry(domain.to_string()).or_insert(len);
        }
        if let Some(owner) = record.get("owner").and_then(|v| v.as_str()) {
            let len = provider_index.len();
            provider_index.entry(owner.to_string()).or_insert(len);
        }
    }

    let n_providers = provider_index.len().max(1);

    let mut inputs: Vec<Vec<f64>> = Vec::with_capacity(records.len());
    let mut targets: Vec<Vec<f64>> = Vec::with_capacity(records.len());

    let max_latency = records
        .iter()
        .filter_map(|r| r.get("latency_ms").and_then(|v| v.as_f64()))
        .fold(1.0_f64, f64::max);
    let max_param_size = records
        .iter()
        .filter_map(|r| r.get("param_size_bytes").and_then(|v| v.as_f64()))
        .fold(1.0_f64, f64::max);
    let max_gate_load = records
        .iter()
        .filter_map(|r| r.get("gate_load").and_then(|v| v.as_f64()))
        .fold(1.0_f64, f64::max);

    for record in records {
        let mut feature = vec![0.0_f64; 36];

        if let Some(method) = record.get("method").and_then(|v| v.as_str()) {
            let domain = method.split('.').next().unwrap_or(method);
            if let Some(&idx) = domain_index.get(domain) {
                if idx < 32 {
                    feature[idx] = 1.0;
                }
            }
        }

        let latency_ms = record
            .get("latency_ms")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let success = record
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        feature[32] = record
            .get("param_size_bytes")
            .and_then(|v| v.as_f64())
            .map_or(0.0, |v| v / max_param_size);
        feature[33] = record
            .get("gate_load")
            .and_then(|v| v.as_f64())
            .map_or(0.0, |v| v / max_gate_load);
        feature[34] = latency_ms / max_latency;
        feature[35] = record
            .get("topology_affinity")
            .and_then(|v| v.as_f64())
            .unwrap_or(if success { 1.0 } else { 0.0 });

        inputs.push(feature);

        let mut target = vec![0.0_f64; n_providers];
        if let Some(owner) = record.get("owner").and_then(|v| v.as_str()) {
            if let Some(&idx) = provider_index.get(owner) {
                let reward = if success {
                    1.0 / (1.0 + latency_ms)
                } else {
                    0.0
                };
                target[idx] = reward;
            }
        }
        targets.push(target);
    }

    let mut mlp = SimpleMlp::from_dims(&[36, n_providers], Activation::Sigmoid);
    let config = TrainConfig {
        learning_rate,
        epochs,
    };

    match mlp.train(&inputs, &targets, &config) {
        Ok(mse) => {
            let trained_layers: Vec<Value> = mlp
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

            let mut domain_list: Vec<(&str, usize)> =
                domain_index.iter().map(|(k, &v)| (k.as_str(), v)).collect();
            domain_list.sort_by_key(|&(_, idx)| idx);

            let mut provider_list: Vec<(&str, usize)> =
                provider_index.iter().map(|(k, &v)| (k.as_str(), v)).collect();
            provider_list.sort_by_key(|&(_, idx)| idx);

            let serialized = if let Some(path) = output_path {
                match mlp.to_json() {
                    Ok(json_str) => match std::fs::write(path, &json_str) {
                        Ok(()) => Some(path),
                        Err(_) => None,
                    },
                    Err(_) => None,
                }
            } else {
                None
            };

            let mut response = serde_json::json!({
                "layers": trained_layers,
                "mse": mse,
                "epochs": epochs,
                "records_processed": records.len(),
                "providers": provider_list.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
                "domains": domain_list.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            });

            if let Some(path) = serialized {
                response["output_path"] = Value::String(path.to_string());
            }

            JsonRpcResponse::success(id, response)
        }
        Err(e) => JsonRpcResponse::error(
            id,
            INTERNAL_ERROR,
            format!("Perceptron training failed: {e}"),
        ),
    }
}

/// `ml.mlp_infer` — Batch inference on dispatch telemetry via trained perceptron.
///
/// Runs forward pass through a trained model (loaded from file or inline weights)
/// on new telemetry vectors. Returns per-record provider scores for routing.
///
/// Wire contract:
/// ```json
/// { "model_path": "/path/to/neural_routing_perceptron.bin",
///   "records": [{"method":"stats.mean","owner":"","latency_ms":1.2,"success":true},...] }
/// ```
/// Or with inline model:
/// ```json
/// { "model": {"layers": [...]},
///   "records": [...] }
/// ```
pub(super) fn ml_mlp_infer(params: &Value, id: Value) -> JsonRpcResponse {
    let mlp = if let Some(path) = params.get("model_path").and_then(|v| v.as_str()) {
        match std::fs::read_to_string(path) {
            Ok(json_str) => match SimpleMlp::from_json(&json_str) {
                Ok(m) => m,
                Err(e) => {
                    return JsonRpcResponse::error(
                        id,
                        INVALID_PARAMS,
                        format!("Failed to parse model at {path}: {e}"),
                    );
                }
            },
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Failed to read model at {path}: {e}"),
                );
            }
        }
    } else if let Some(model_val) = params.get("model") {
        match serde_json::from_value::<SimpleMlp>(model_val.clone()) {
            Ok(m) => m,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Failed to parse inline model: {e}"),
                );
            }
        }
    } else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: model_path (string) or model (object)",
        );
    };

    let Some(records) = params.get("records").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: records (array of telemetry objects)",
        );
    };
    if records.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "records must be non-empty");
    }

    let domain_index = params
        .get("domain_index")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_u64().map(|i| (k.clone(), i as usize)))
                .collect::<HashMap<String, usize>>()
        });

    let providers = params
        .get("providers")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<String>>()
        });

    let max_latency = records
        .iter()
        .filter_map(|r| r.get("latency_ms").and_then(|v| v.as_f64()))
        .fold(1.0_f64, f64::max);

    let mut results: Vec<Value> = Vec::with_capacity(records.len());

    for record in records {
        let feature = extract_telemetry_feature(record, domain_index.as_ref(), max_latency);
        let scores = mlp.forward(&feature);

        let best_idx = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i);

        let mut entry = serde_json::json!({
            "scores": scores,
            "best_index": best_idx,
        });

        if let Some(ref prov) = providers {
            if best_idx < prov.len() {
                entry["best_provider"] = Value::String(prov[best_idx].clone());
            }
        }

        results.push(entry);
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "results": results,
            "records_processed": records.len(),
        }),
    )
}

/// `ml.mlp_save` — Persist a trained model to disk.
///
/// Serializes the model as JSON and writes to the specified path.
/// The path must be within allowed directories (no traversal).
///
/// Wire contract:
/// ```json
/// { "model": {"layers": [...]}, "path": "/data/gate/neural_routing_perceptron.bin" }
/// ```
pub(super) fn ml_mlp_save(params: &Value, id: Value) -> JsonRpcResponse {
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
pub(super) fn ml_mlp_load(params: &Value, id: Value) -> JsonRpcResponse {
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

fn extract_telemetry_feature(
    record: &Value,
    domain_index: Option<&HashMap<String, usize>>,
    max_latency: f64,
) -> Vec<f64> {
    let mut feature = vec![0.0_f64; 36];

    if let Some(method) = record.get("method").and_then(|v| v.as_str()) {
        let domain = method.split('.').next().unwrap_or(method);
        if let Some(idx) = domain_index.and_then(|di| di.get(domain)).copied() {
            if idx < 32 {
                feature[idx] = 1.0;
            }
        }
    }

    let latency_ms = record
        .get("latency_ms")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let success = record
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    feature[32] = record
        .get("param_size_bytes")
        .and_then(|v| v.as_f64())
        .map_or(0.0, |v| (v / 1_000_000.0).min(1.0));
    feature[33] = record
        .get("gate_load")
        .and_then(|v| v.as_f64())
        .map_or(0.0, |v| v.min(1.0));
    feature[34] = latency_ms / max_latency;
    feature[35] = record
        .get("topology_affinity")
        .and_then(|v| v.as_f64())
        .unwrap_or(if success { 1.0 } else { 0.0 });

    feature
}
