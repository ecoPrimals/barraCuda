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
    use crate::health::PrimalHealth;
    match primal.health_check().await {
        Ok(report) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "name": report.name,
                "version": report.version,
                "status": format!("{}", report.status),
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
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let mut results = Vec::new();

    // Matmul validation: 4x4 identity
    let matmul_pass = {
        let data: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        match (
            barracuda::tensor::Tensor::from_data(&data, vec![4, 4], dev.clone()),
            barracuda::tensor::Tensor::from_data(&input, vec![4, 4], dev.clone()),
        ) {
            (Ok(eye), Ok(inp)) => inp
                .matmul(&eye)
                .and_then(|out| out.to_vec())
                .is_ok_and(|v| v.iter().zip(&input).all(|(a, b)| (a - b).abs() < 1e-4)),
            _ => false,
        }
    };
    results.push(serde_json::json!({"test": "matmul_identity", "pass": matmul_pass}));

    // Tensor round-trip validation
    let roundtrip_pass = {
        let data = vec![std::f32::consts::PI, 2.71, 1.41, 0.57];
        barracuda::tensor::Tensor::from_data(&data, vec![4], dev.clone())
            .and_then(|t| t.to_vec())
            .is_ok_and(|v| v.iter().zip(&data).all(|(a, b)| (a - b).abs() < 1e-6))
    };
    results.push(serde_json::json!({"test": "tensor_roundtrip", "pass": roundtrip_pass}));

    let all_pass = results.iter().all(|r| r["pass"].as_bool().unwrap_or(false));

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "gpu_available": true,
            "status": if all_pass { "pass" } else { "partial_fail" },
            "tests": results,
        }),
    )
}

/// `barracuda.compute.dispatch` — Dispatch a named compute operation.
///
/// Rather than accepting raw WGSL (which would require shader security auditing),
/// this dispatches named operations from barraCuda's shader library. Pass input
/// data and the operation produces output stored in the tensor store.
async fn compute_dispatch(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let Some(op) = params.get("op").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: op (e.g. 'zeros', 'ones', 'from_data')",
        );
    };

    match op {
        "zeros" => {
            let Some(shape) = params.get("shape").and_then(|v| v.as_array()) else {
                return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shape");
            };
            let shape_vec: Vec<usize> = shape
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            match barracuda::tensor::Tensor::zeros_on(shape_vec.clone(), dev.clone()).await {
                Ok(t) => {
                    let tensor_id = primal.store_tensor(t);
                    JsonRpcResponse::success(
                        id,
                        serde_json::json!({
                            "status": "completed", "op": "zeros", "tensor_id": tensor_id, "shape": shape_vec,
                        }),
                    )
                }
                Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("zeros failed: {e}")),
            }
        }
        "ones" => {
            let Some(shape) = params.get("shape").and_then(|v| v.as_array()) else {
                return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shape");
            };
            let shape_vec: Vec<usize> = shape
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect();
            match barracuda::tensor::Tensor::ones_on(shape_vec.clone(), dev.clone()).await {
                Ok(t) => {
                    let tensor_id = primal.store_tensor(t);
                    JsonRpcResponse::success(
                        id,
                        serde_json::json!({
                            "status": "completed", "op": "ones", "tensor_id": tensor_id, "shape": shape_vec,
                        }),
                    )
                }
                Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("ones failed: {e}")),
            }
        }
        "read" => {
            let Some(tensor_id) = params.get("tensor_id").and_then(|v| v.as_str()) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    "Missing required param: tensor_id",
                );
            };
            let Some(tensor) = primal.get_tensor(tensor_id) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Tensor not found: {tensor_id}"),
                );
            };
            match tensor.to_vec() {
                Ok(data) => JsonRpcResponse::success(
                    id,
                    serde_json::json!({
                        "status": "completed", "tensor_id": tensor_id,
                        "shape": tensor.shape(), "data": data,
                    }),
                ),
                Err(e) => {
                    JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Readback failed: {e}"))
                }
            }
        }
        _ => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Unknown op: {op}. Available: zeros, ones, read"),
        ),
    }
}

/// `barracuda.tensor.create` — Create a real tensor on the GPU device.
async fn tensor_create(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let Some(shape) = params.get("shape").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shape");
    };

    let shape_vec: Vec<usize> = shape
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();

    let elements: usize = shape_vec.iter().product();

    // If data provided, use it; otherwise create zeros
    let data: Vec<f32> = if let Some(arr) = params.get("data").and_then(|v| v.as_array()) {
        arr.iter()
            .filter_map(|v| v.as_f64().map(|n| n as f32))
            .collect()
    } else {
        vec![0.0; elements]
    };

    if data.len() != elements {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!(
                "data length {} does not match shape product {elements}",
                data.len()
            ),
        );
    }

    match barracuda::tensor::Tensor::from_data(&data, shape_vec.clone(), dev.clone()) {
        Ok(tensor) => {
            let tensor_id = primal.store_tensor(tensor);
            JsonRpcResponse::success(
                id,
                serde_json::json!({
                    "tensor_id": tensor_id,
                    "shape": shape_vec,
                    "elements": elements,
                    "dtype": "f32",
                }),
            )
        }
        Err(e) => {
            JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Tensor creation failed: {e}"))
        }
    }
}

/// `barracuda.tensor.matmul` — Matrix multiply two stored tensors.
async fn tensor_matmul(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    if primal.device().is_none() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    }

    let Some(lhs_str) = params.get("lhs_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: lhs_id");
    };
    let Some(rhs_str) = params.get("rhs_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: rhs_id");
    };

    let Some(lhs) = primal.get_tensor(lhs_str) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {lhs_str}"));
    };
    let Some(rhs) = primal.get_tensor(rhs_str) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {rhs_str}"));
    };

    match lhs.matmul_ref(&rhs) {
        Ok(result) => {
            let shape: Vec<usize> = result.shape().to_vec();
            let elements = shape.iter().product::<usize>();
            let result_id = primal.store_tensor(result);
            JsonRpcResponse::success(
                id,
                serde_json::json!({
                    "status": "completed",
                    "result_id": result_id,
                    "shape": shape,
                    "elements": elements,
                }),
            )
        }
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Matmul failed: {e}")),
    }
}

/// `barracuda.fhe.ntt` — Execute Number Theoretic Transform on GPU.
async fn fhe_ntt(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let Some(modulus) = params.get("modulus").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: modulus");
    };
    let Some(degree) = params.get("degree").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: degree");
    };
    let Some(root_of_unity) = params.get("root_of_unity").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: root_of_unity");
    };
    let Some(coefficients) = params.get("coefficients").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: coefficients");
    };

    let poly: Vec<u64> = coefficients.iter().filter_map(|v| v.as_u64()).collect();
    if poly.len() != degree as usize {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("coefficients length {} != degree {degree}", poly.len()),
        );
    }

    // u64 → u32 pairs → f32 bit patterns → Tensor
    let u32_pairs: Vec<u32> = poly
        .iter()
        .flat_map(|&x| [(x & 0xFFFF_FFFF) as u32, (x >> 32) as u32])
        .collect();
    let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();

    let input_tensor =
        match barracuda::tensor::Tensor::from_data(&f32_bits, vec![poly.len() * 2], dev.clone()) {
            Ok(t) => t,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("Tensor creation failed: {e}"),
                )
            }
        };

    let ntt = match barracuda::ops::fhe_ntt::FheNtt::new(
        input_tensor,
        degree as u32,
        modulus,
        root_of_unity,
    ) {
        Ok(n) => n,
        Err(e) => {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, format!("NTT setup failed: {e}"))
        }
    };

    match ntt.execute() {
        Ok(result_tensor) => {
            // Read back and convert u32 pairs → u64
            match result_tensor.to_vec_u32() {
                Ok(u32_data) => {
                    let result_u64: Vec<u64> = u32_data
                        .chunks(2)
                        .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
                        .collect();
                    JsonRpcResponse::success(
                        id,
                        serde_json::json!({
                            "status": "completed",
                            "modulus": modulus,
                            "degree": degree,
                            "result": result_u64,
                        }),
                    )
                }
                Err(e) => {
                    JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Readback failed: {e}"))
                }
            }
        }
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("NTT execution failed: {e}")),
    }
}

/// `barracuda.fhe.pointwise_mul` — Execute pointwise polynomial multiplication on GPU.
async fn fhe_pointwise_mul(primal: &BarraCudaPrimal, params: &Value, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let Some(modulus) = params.get("modulus").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: modulus");
    };
    let Some(degree) = params.get("degree").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: degree");
    };
    let Some(a_coeffs) = params.get("a").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: a");
    };
    let Some(b_coeffs) = params.get("b").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: b");
    };

    let a: Vec<u64> = a_coeffs.iter().filter_map(|v| v.as_u64()).collect();
    let b: Vec<u64> = b_coeffs.iter().filter_map(|v| v.as_u64()).collect();

    if a.len() != degree as usize || b.len() != degree as usize {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("coefficient arrays must have {degree} elements"),
        );
    }

    let to_tensor = |poly: &[u64]| -> Result<barracuda::tensor::Tensor, String> {
        let u32_pairs: Vec<u32> = poly
            .iter()
            .flat_map(|&x| [(x & 0xFFFF_FFFF) as u32, (x >> 32) as u32])
            .collect();
        let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();
        barracuda::tensor::Tensor::from_data(&f32_bits, vec![poly.len() * 2], dev.clone())
            .map_err(|e| e.to_string())
    };

    let (a_tensor, b_tensor) = match (to_tensor(&a), to_tensor(&b)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Tensor creation failed: {e}"),
            )
        }
    };

    let op = match barracuda::ops::fhe_pointwise_mul::FhePointwiseMul::new(
        a_tensor,
        b_tensor,
        degree as u32,
        modulus,
    ) {
        Ok(op) => op,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Pointwise mul setup failed: {e}"),
            )
        }
    };

    match op.execute() {
        Ok(result_tensor) => match result_tensor.to_vec_u32() {
            Ok(u32_data) => {
                let result_u64: Vec<u64> = u32_data
                    .chunks(2)
                    .map(|c| u64::from(c[0]) | (u64::from(c[1]) << 32))
                    .collect();
                JsonRpcResponse::success(
                    id,
                    serde_json::json!({
                        "status": "completed",
                        "modulus": modulus,
                        "degree": degree,
                        "result": result_u64,
                    }),
                )
            }
            Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Readback failed: {e}")),
        },
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Pointwise mul failed: {e}")),
    }
}

#[cfg(test)]
#[path = "methods_tests.rs"]
mod tests;
