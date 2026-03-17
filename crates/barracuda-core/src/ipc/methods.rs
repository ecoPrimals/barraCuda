// SPDX-License-Identifier: AGPL-3.0-only
//! JSON-RPC method handlers for barraCuda IPC endpoints.
//!
//! Each handler follows the semantic naming standard from wateringHole:
//! `{namespace}.{domain}.{operation}`. Method names are derived from
//! [`PRIMAL_NAMESPACE`](crate::PRIMAL_NAMESPACE) at startup — no
//! hardcoded primal names on the wire.

use super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse, METHOD_NOT_FOUND};
use crate::BarraCudaPrimal;
use serde_json::Value;
use std::sync::LazyLock;

/// Domain-qualified method suffixes (namespace-agnostic).
///
/// Single source of truth for which operations this primal supports.
/// Full wire names are built at startup by prepending [`PRIMAL_NAMESPACE`](crate::PRIMAL_NAMESPACE).
const METHOD_SUFFIXES: &[&str] = &[
    "primal.info",
    "primal.capabilities",
    "device.list",
    "device.probe",
    "health.check",
    "tolerances.get",
    "validate.gpu_stack",
    "compute.dispatch",
    "tensor.create",
    "tensor.matmul",
    "fhe.ntt",
    "fhe.pointwise_mul",
];

/// All JSON-RPC method names this primal supports, fully qualified.
///
/// Built from [`PRIMAL_NAMESPACE`](crate::PRIMAL_NAMESPACE) + `METHOD_SUFFIXES`
/// so renaming the namespace automatically updates all wire names.
pub static REGISTERED_METHODS: LazyLock<Vec<String>> = LazyLock::new(|| {
    METHOD_SUFFIXES
        .iter()
        .map(|suffix| format!("{}.{suffix}", crate::PRIMAL_NAMESPACE))
        .collect()
});

/// Strip the namespace prefix from a fully-qualified method name.
pub fn method_suffix(method: &str) -> Option<&str> {
    method
        .strip_prefix(crate::PRIMAL_NAMESPACE)
        .and_then(|s| s.strip_prefix('.'))
}

/// Route a JSON-RPC method call to the appropriate handler.
pub async fn dispatch(
    primal: &BarraCudaPrimal,
    method: &str,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    match method_suffix(method) {
        Some("primal.info") => primal_info(primal, id),
        Some("primal.capabilities") => primal_capabilities(primal, id),
        Some("device.list") => device_list(primal, id).await,
        Some("device.probe") => device_probe(primal, id).await,
        Some("health.check") => health_check(primal, id).await,
        Some("tolerances.get") => tolerances_get(params, id),
        Some("validate.gpu_stack") => validate_gpu_stack(primal, id).await,
        Some("compute.dispatch") => compute_dispatch(primal, params, id).await,
        Some("tensor.create") => tensor_create(primal, params, id).await,
        Some("tensor.matmul") => tensor_matmul(primal, params, id).await,
        Some("fhe.ntt") => fhe_ntt(primal, params, id).await,
        Some("fhe.pointwise_mul") => fhe_pointwise_mul(primal, params, id).await,
        _ => JsonRpcResponse::error(id, METHOD_NOT_FOUND, format!("Unknown method: {method}")),
    }
}

/// `barracuda.primal.info` — Primal identity for runtime discovery.
///
/// Other primals call this method (not hardcoded names) to identify barraCuda.
fn primal_info(_primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": crate::PRIMAL_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "protocol": "json-rpc-2.0",
            "namespace": crate::PRIMAL_NAMESPACE,
            "license": "AGPL-3.0-only",
        }),
    )
}

/// `barracuda.primal.capabilities` — Advertise capabilities for discovery.
///
/// Returns the set of capabilities this primal provides at runtime. Other
/// primals use this to discover what barraCuda can do (capability-based
/// routing) rather than relying on hardcoded primal names.
fn primal_capabilities(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let has_gpu = primal.device().is_some();
    let has_f64 = primal.device().is_some_and(|d| d.has_f64_shaders());
    let has_spirv = primal.device().is_some_and(|d| d.has_spirv_passthrough());

    let version = env!("CARGO_PKG_VERSION");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "provides": [
                { "id": "gpu.compute", "version": version },
                { "id": "tensor.ops", "version": version },
                { "id": "gpu.dispatch", "version": version },
            ],
            "requires": [
                { "id": "shader.compile", "version": ">=0.1.0", "optional": true },
            ],
            "domains": [
                "gpu_compute",
                "tensor_ops",
                "fhe",
                "molecular_dynamics",
                "lattice_qcd",
                "statistics",
                "hydrology",
                "bio",
            ],
            "methods": &*REGISTERED_METHODS,
            "hardware": {
                "gpu_available": has_gpu,
                "f64_shaders": has_f64,
                "spirv_passthrough": has_spirv,
            },
        }),
    )
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
                "status": report.status.to_string(),
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
///
/// Uses 2×2 identity matrix: minimal size that validates matmul path without
/// unnecessary GPU memory/transfer overhead. Aligned with tarpc validate_gpu_stack.
async fn validate_gpu_stack(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let mut results = Vec::new();

    // Matmul validation: 2×2 identity (minimal, fast, validates GPU stack)
    let dev_arc = std::sync::Arc::new(dev);
    let matmul_pass = {
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        match (
            barracuda::tensor::Tensor::from_data(&eye, vec![2, 2], dev_arc.clone()),
            barracuda::tensor::Tensor::from_data(&input, vec![2, 2], dev_arc.clone()),
        ) {
            (Ok(eye_t), Ok(inp)) => inp
                .matmul(&eye_t)
                .and_then(|out| out.to_vec())
                .is_ok_and(|v| v.iter().zip(&input).all(|(a, b)| (a - b).abs() < 1e-4)),
            _ => false,
        }
    };
    results.push(serde_json::json!({"test": "matmul_identity", "pass": matmul_pass}));

    // Tensor round-trip validation
    let roundtrip_pass = {
        let data = vec![std::f32::consts::PI, 2.71, 1.41, 0.57];
        barracuda::tensor::Tensor::from_data(&data, vec![4], dev_arc)
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
/// Parse a JSON array of u64 values into a `Vec<usize>`, returning `None` if
/// any dimension overflows the platform `usize`.
fn parse_shape(arr: &[Value]) -> Option<Vec<usize>> {
    arr.iter()
        .filter_map(|v| v.as_u64())
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()
}

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
        "zeros" | "ones" => {
            let Some(shape) = params.get("shape").and_then(|v| v.as_array()) else {
                return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: shape");
            };
            let Some(shape_vec) = parse_shape(shape) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    "Shape dimension exceeds platform usize",
                );
            };
            let dev_arc = std::sync::Arc::new(dev);
            let result = if op == "zeros" {
                barracuda::tensor::Tensor::zeros_on(shape_vec.clone(), dev_arc).await
            } else {
                barracuda::tensor::Tensor::ones_on(shape_vec.clone(), dev_arc).await
            };
            match result {
                Ok(t) => {
                    let tensor_id = primal.store_tensor(t);
                    JsonRpcResponse::success(
                        id,
                        serde_json::json!({
                            "status": "completed", "op": op, "tensor_id": tensor_id, "shape": shape_vec,
                        }),
                    )
                }
                Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("{op} failed: {e}")),
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

    let Some(shape_vec) = parse_shape(shape) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Shape dimension exceeds platform usize",
        );
    };

    let elements: usize = shape_vec.iter().product();

    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
    )]
    let data: Vec<f32> = params.get("data").and_then(|v| v.as_array()).map_or_else(
        || vec![0.0; elements],
        |arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|n| n as f32))
                .collect()
        },
    );

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

    match barracuda::tensor::Tensor::from_data(&data, shape_vec.clone(), std::sync::Arc::new(dev)) {
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
    let Ok(degree_usize) = usize::try_from(degree) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("degree {degree} too large"));
    };
    let Ok(degree_u32) = u32::try_from(degree) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("degree {degree} exceeds u32::MAX"),
        );
    };
    if poly.len() != degree_usize {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("coefficients length {} != degree {degree}", poly.len()),
        );
    }

    // u64 → u32 pairs → f32 bit patterns → Tensor
    #[expect(
        clippy::cast_possible_truncation,
        reason = "intentional u64→u32 split for FHE coefficient layout"
    )]
    let u32_pairs: Vec<u32> = poly
        .iter()
        .flat_map(|&x| [x as u32, (x >> 32) as u32])
        .collect();
    let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();

    let input_tensor = match barracuda::tensor::Tensor::from_data(
        &f32_bits,
        vec![poly.len() * 2],
        std::sync::Arc::new(dev),
    ) {
        Ok(t) => t,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Tensor creation failed: {e}"),
            );
        }
    };

    let ntt = match barracuda::ops::fhe_ntt::FheNtt::new(
        input_tensor,
        degree_u32,
        modulus,
        root_of_unity,
    ) {
        Ok(n) => n,
        Err(e) => {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, format!("NTT setup failed: {e}"));
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

    let Ok(degree_usize) = usize::try_from(degree) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("degree {degree} too large"));
    };
    let Ok(degree_u32) = u32::try_from(degree) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("degree {degree} exceeds u32::MAX"),
        );
    };
    if a.len() != degree_usize || b.len() != degree_usize {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("coefficient arrays must have {degree} elements"),
        );
    }

    let to_tensor = |poly: &[u64]| -> barracuda::error::Result<barracuda::tensor::Tensor> {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "intentional u64→u32 split for FHE coefficient layout"
        )]
        let u32_pairs: Vec<u32> = poly
            .iter()
            .flat_map(|&x| [x as u32, (x >> 32) as u32])
            .collect();
        let f32_bits: Vec<f32> = u32_pairs.iter().map(|&x| f32::from_bits(x)).collect();
        barracuda::tensor::Tensor::from_data(
            &f32_bits,
            vec![poly.len() * 2],
            std::sync::Arc::new(dev.clone()),
        )
    };

    let (a_tensor, b_tensor) = match (to_tensor(&a), to_tensor(&b)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Tensor creation failed: {e}"),
            );
        }
    };

    let op = match barracuda::ops::fhe_pointwise_mul::FhePointwiseMul::new(
        a_tensor, b_tensor, degree_u32, modulus,
    ) {
        Ok(op) => op,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Pointwise mul setup failed: {e}"),
            );
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
