// SPDX-License-Identifier: AGPL-3.0-or-later
//! Batch ML inference: run trained perceptron forward pass on telemetry vectors.

use super::super::super::jsonrpc::{INVALID_PARAMS, JsonRpcResponse};
use barracuda::nn::simple_mlp::SimpleMlp;
use serde_json::Value;
use std::collections::HashMap;

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
pub(in crate::ipc::methods) fn ml_mlp_infer(params: &Value, id: Value) -> JsonRpcResponse {
    let mlp = if let Some(path) = params.get("model_path").and_then(|v| v.as_str()) {
        match std::fs::read(path) {
            Ok(data) => match SimpleMlp::from_auto(&data) {
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

        if let Some(ref prov) = providers
            && best_idx < prov.len()
        {
            entry["best_provider"] = Value::String(prov[best_idx].clone());
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

/// Extract a 36-dim feature vector from a single telemetry record.
///
/// Feature layout: slots 0..32 are one-hot domain encoding (from `domain_index`
/// map), slot 32 is normalized param_size_bytes, 33 is gate_load, 34 is
/// latency_ms, and 35 is topology_affinity (or binary success signal).
pub(super) fn extract_telemetry_feature(
    record: &Value,
    domain_index: Option<&HashMap<String, usize>>,
    max_latency: f64,
) -> Vec<f64> {
    let mut feature = vec![0.0_f64; 36];

    if let Some(method) = record.get("method").and_then(|v| v.as_str()) {
        let domain = method.split('.').next().unwrap_or(method);
        if let Some(idx) = domain_index.and_then(|di| di.get(domain)).copied()
            && idx < 32
        {
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
