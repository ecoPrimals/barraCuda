// SPDX-License-Identifier: AGPL-3.0-or-later
//! ML inference handlers for JSON-RPC IPC.
//!
//! Inline-data CPU paths for lightweight ML operations suitable for
//! composition graph nodes. GPU tensor ops live in `tensor.rs`.

use super::super::jsonrpc::{INVALID_PARAMS, JsonRpcResponse};
use barracuda::nn::simple_mlp::{Activation, DenseLayer, SimpleMlp};
use serde_json::Value;

fn extract_f64_array(params: &Value, key: &str) -> Option<Vec<f64>> {
    params
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
}

fn extract_matrix(params: &Value, key: &str) -> Option<Vec<Vec<f64>>> {
    params.get(key).and_then(|v| v.as_array()).map(|rows| {
        rows.iter()
            .filter_map(|row| {
                row.as_array()
                    .map(|cols| cols.iter().filter_map(|c| c.as_f64()).collect())
            })
            .collect()
    })
}

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
