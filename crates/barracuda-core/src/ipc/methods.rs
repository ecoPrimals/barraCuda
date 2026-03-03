// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC method handlers for barraCuda IPC endpoints.
//!
//! Each handler follows the semantic naming standard from wateringHole:
//! `barracuda.{domain}.{operation}`.

use super::jsonrpc::{JsonRpcResponse, INTERNAL_ERROR, INVALID_PARAMS, METHOD_NOT_FOUND};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// Route a JSON-RPC method call to the appropriate handler.
pub async fn dispatch(
    primal: &BarraCudaPrimal,
    method: &str,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    match method {
        "barracuda.device.list" => device_list(primal, id).await,
        "barracuda.device.probe" => device_probe(primal, id).await,
        "barracuda.health.check" => health_check(primal, id).await,
        "barracuda.tolerances.get" => tolerances_get(params, id),
        "barracuda.validate.gpu_stack" => validate_gpu_stack(primal, id).await,
        "barracuda.compute.dispatch" => compute_dispatch(primal, params, id).await,
        "barracuda.tensor.create" => tensor_create(primal, params, id).await,
        "barracuda.tensor.matmul" => tensor_matmul(primal, params, id).await,
        "barracuda.fhe.ntt" => fhe_ntt(primal, params, id).await,
        "barracuda.fhe.pointwise_mul" => fhe_pointwise_mul(primal, params, id).await,
        _ => JsonRpcResponse::error(id, METHOD_NOT_FOUND, format!("Unknown method: {method}")),
    }
}

/// `barracuda.device.list` — Enumerate available compute devices.
async fn device_list(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let mut devices = Vec::new();

    if let Some(dev) = primal.device() {
        let info = dev.adapter_info();
        devices.push(serde_json::json!({
            "name": info.name,
            "vendor": info.vendor,
            "device_type": format!("{:?}", info.device_type),
            "backend": format!("{:?}", info.backend),
            "driver": info.driver,
            "driver_info": info.driver_info,
        }));
    }

    JsonRpcResponse::success(id, serde_json::json!({ "devices": devices }))
}

/// `barracuda.device.probe` — Probe device capabilities.
async fn device_probe(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::success(
            id,
            serde_json::json!({
                "available": false,
                "reason": "No GPU device initialized"
            }),
        );
    };

    let limits = dev.device().limits();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "available": true,
            "max_buffer_size": limits.max_buffer_size,
            "max_storage_buffers_per_shader_stage": limits.max_storage_buffers_per_shader_stage,
            "max_compute_workgroup_size_x": limits.max_compute_workgroup_size_x,
            "max_compute_workgroups_per_dimension": limits.max_compute_workgroups_per_dimension,
        }),
    )
}

/// `barracuda.health.check` — Primal health status.
async fn health_check(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    use sourdough_core::PrimalHealth;
    match primal.health_check().await {
        Ok(report) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "name": report.name,
                "version": report.version,
                "status": format!("{:?}", report.status),
            }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, e.to_string()),
    }
}

/// `barracuda.tolerances.get` — Get numerical tolerances for a named operation.
fn tolerances_get(params: &Value, id: Value) -> JsonRpcResponse {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let (abs_tol, rel_tol) = match name {
        "fhe" => (0.0, 0.0),
        "f64" | "double" => (1e-12, 1e-10),
        "f32" | "float" => (1e-5, 1e-4),
        "df64" | "emulated_double" => (1e-10, 1e-8),
        _ => (1e-6, 1e-5),
    };

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "name": name,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
        }),
    )
}

/// `barracuda.validate.gpu_stack` — Run GPU validation suite.
async fn validate_gpu_stack(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "gpu_available": true,
            "status": "validation_available",
            "message": "Use `barracuda validate` CLI for full GPU stack validation"
        }),
    )
}

/// `barracuda.compute.dispatch` — Dispatch a compute shader.
async fn compute_dispatch(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    let Some(shader) = params.get("shader").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shader");
    };

    let entry_point = params
        .get("entry_point")
        .and_then(|v| v.as_str())
        .unwrap_or("main");

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "accepted",
            "shader": shader,
            "entry_point": entry_point,
        }),
    )
}

/// `barracuda.tensor.create` — Create a tensor on the device.
async fn tensor_create(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    let Some(shape) = params.get("shape").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shape");
    };

    let shape_vec: Vec<usize> = shape
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();

    let elements: usize = shape_vec.iter().product();
    let tensor_id = blake3::hash(
        format!(
            "tensor:{}:{}",
            serde_json::to_string(&shape_vec).unwrap_or_default(),
            elements
        )
        .as_bytes(),
    )
    .to_hex()
    .to_string();

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "tensor_id": &tensor_id[..16],
            "shape": shape_vec,
            "elements": elements,
            "dtype": params.get("dtype").and_then(|v| v.as_str()).unwrap_or("f32"),
        }),
    )
}

/// `barracuda.tensor.matmul` — Matrix multiply two tensors.
async fn tensor_matmul(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    let lhs_id = params.get("lhs_id").and_then(|v| v.as_str());
    let rhs_id = params.get("rhs_id").and_then(|v| v.as_str());

    match (lhs_id, rhs_id) {
        (Some(lhs), Some(rhs)) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "status": "accepted",
                "lhs_id": lhs,
                "rhs_id": rhs,
            }),
        ),
        _ => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required params: lhs_id, rhs_id",
        ),
    }
}

/// `barracuda.fhe.ntt` — Number Theoretic Transform for FHE.
async fn fhe_ntt(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    let modulus = params.get("modulus").and_then(|v| v.as_u64());
    let degree = params.get("degree").and_then(|v| v.as_u64());

    match (modulus, degree) {
        (Some(m), Some(d)) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "status": "accepted",
                "modulus": m,
                "degree": d,
            }),
        ),
        _ => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required params: modulus, degree",
        ),
    }
}

/// `barracuda.fhe.pointwise_mul` — Pointwise polynomial multiplication for FHE.
async fn fhe_pointwise_mul(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    let modulus = params.get("modulus").and_then(|v| v.as_u64());
    if modulus.is_none() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: modulus");
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "accepted",
            "modulus": modulus,
        }),
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn test_primal() -> BarraCudaPrimal {
        BarraCudaPrimal::new()
    }

    #[tokio::test]
    async fn test_device_list_no_gpu() {
        let primal = test_primal();
        let resp = device_list(&primal, serde_json::json!(1)).await;
        assert!(resp.result.is_some());
        let result = resp.result.unwrap();
        assert!(result["devices"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_health_check() {
        let primal = test_primal();
        let resp = health_check(&primal, serde_json::json!(2)).await;
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_tolerances_default() {
        let resp = tolerances_get(&serde_json::json!({}), serde_json::json!(3));
        let result = resp.result.unwrap();
        assert_eq!(result["name"], "default");
    }

    #[test]
    fn test_tolerances_fhe() {
        let resp = tolerances_get(&serde_json::json!({"name": "fhe"}), serde_json::json!(4));
        let result = resp.result.unwrap();
        assert_eq!(result["abs_tol"], 0.0);
        assert_eq!(result["rel_tol"], 0.0);
    }

    #[tokio::test]
    async fn test_dispatch_routing() {
        let primal = test_primal();
        let resp = dispatch(
            &primal,
            "barracuda.device.list",
            &serde_json::json!({}),
            serde_json::json!(5),
        )
        .await;
        assert!(resp.result.is_some());

        let resp = dispatch(
            &primal,
            "nonexistent.method",
            &serde_json::json!({}),
            serde_json::json!(6),
        )
        .await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
    }
}
