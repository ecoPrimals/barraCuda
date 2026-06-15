// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML training handlers: general MLP training and perceptron pipeline.

use super::super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::super::params::{extract_f64_array, extract_matrix};
use super::parse_activation;
use barracuda::nn::simple_mlp::{Activation, DenseLayer, SimpleMlp, TrainConfig};
use serde_json::Value;
use std::collections::HashMap;

/// `ml.mlp_train` — inline-data MLP training via backpropagation (CPU).
///
/// Supports two forms for the `layers` parameter:
/// - **Shorthand (dims)**: `[36, 16]` — creates a network with Xavier-init random weights.
///   Optional top-level `"activation"` (default `"sigmoid"`) sets hidden activations.
/// - **Explicit (weights)**: `[{"weights":..., "biases":..., "activation":...}]` — resumes
///   training from existing weights (original contract).
pub(in crate::ipc::methods) fn ml_mlp_train(params: &Value, id: Value) -> JsonRpcResponse {
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
pub(in crate::ipc::methods) fn ml_perceptron_train(params: &Value, id: Value) -> JsonRpcResponse {
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
    let epochs = params.get("epochs").and_then(|v| v.as_u64()).unwrap_or(10) as usize;
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
            if let Some(&idx) = domain_index.get(domain)
                && idx < 32 {
                    feature[idx] = 1.0;
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
        if let Some(owner) = record.get("owner").and_then(|v| v.as_str())
            && let Some(&idx) = provider_index.get(owner) {
                let reward = if success {
                    1.0 / (1.0 + latency_ms)
                } else {
                    0.0
                };
                target[idx] = reward;
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

            let mut provider_list: Vec<(&str, usize)> = provider_index
                .iter()
                .map(|(k, &v)| (k.as_str(), v))
                .collect();
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
