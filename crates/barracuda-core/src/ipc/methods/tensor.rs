// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tensor handlers — creation, matmul, and element-wise IPC ops.
//!
//! Wired for ludoSpring composition: each op is a graph node consumable
//! via `tensor.*` JSON-RPC methods per `SEMANTIC_METHOD_NAMING_STANDARD.md`.
//!
//! ## Response Schema (Composition Wire Contract)
//!
//! All `tensor.*` methods return a **standardized** JSON-RPC success payload
//! so that spring typed extractors (`call_extract_f64`, `call_extract_vec_f64`)
//! can rely on consistent keys without per-method guessing.
//!
//! **Tensor-producing ops** (create, matmul, add, scale, clamp, sigmoid):
//! ```json
//! {"status": "completed", "result_id": "t_...", "shape": [M, N], "elements": N}
//! ```
//! `tensor.create` additionally includes `"tensor_id"` (alias of `result_id`)
//! and `"dtype"` for creation-specific metadata.
//!
//! **Scalar-producing ops** (reduce):
//! ```json
//! {"status": "completed", "value": 42.0, "op": "sum"}
//! ```
//! The scalar lives under `"value"` (never `"result"` or `"data"`).

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::compute::parse_shape;
use crate::{BarraCudaPrimal, CpuTensor};
use serde_json::Value;

/// `barracuda.tensor.create` — Create a real tensor on the GPU device.
///
/// Validates shape and data before checking device availability.
pub(super) async fn tensor_create(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
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

    if let Some(dev) = primal.device() {
        match barracuda::tensor::Tensor::from_data(&data, shape_vec.clone(), dev) {
            Ok(tensor) => {
                let tensor_id = primal.store_tensor(tensor);
                JsonRpcResponse::success(
                    id,
                    serde_json::json!({
                        "status": "completed",
                        "tensor_id": &tensor_id,
                        "result_id": tensor_id,
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
    } else {
        let cpu_tensor = CpuTensor {
            data,
            shape: shape_vec.clone(),
        };
        let tensor_id = primal.store_cpu_tensor(cpu_tensor);
        JsonRpcResponse::success(
            id,
            serde_json::json!({
                "status": "completed",
                "tensor_id": &tensor_id,
                "result_id": tensor_id,
                "shape": shape_vec,
                "elements": elements,
                "dtype": "f32",
                "backend": "cpu",
            }),
        )
    }
}

/// `barracuda.tensor.matmul` — Matrix multiply two stored tensors.
///
/// Validates params and tensor existence before attempting computation.
pub(super) async fn tensor_matmul(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(lhs_str) = params.get("lhs_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: lhs_id");
    };
    let Some(rhs_str) = params.get("rhs_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: rhs_id");
    };

    if let Some(lhs) = primal.get_tensor(lhs_str) {
        let Some(rhs) = primal.get_tensor(rhs_str) else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Tensor not found: {rhs_str}"),
            );
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
    } else if let Some(lhs_cpu) = primal.get_cpu_tensor(lhs_str) {
        let Some(rhs_cpu) = primal.get_cpu_tensor(rhs_str) else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Tensor not found: {rhs_str}"),
            );
        };
        cpu_matmul(primal, &lhs_cpu, &rhs_cpu, id)
    } else {
        JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {lhs_str}"))
    }
}

/// `tensor.add` — element-wise add of two tensors or tensor + scalar.
///
/// Accepts `{"tensor_id": "...", "other_id": "..."}` for tensor-tensor add,
/// or `{"tensor_id": "...", "scalar": 1.0}` for broadcast scalar add.
/// Falls back to CPU when tensors are CPU-resident.
pub(super) async fn tensor_add(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };

    if let Some(tensor) = primal.get_tensor(tid) {
        let result = if let Some(scalar) = params.get("scalar").and_then(|v| v.as_f64()) {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
            )]
            tensor.add_scalar(scalar as f32)
        } else if let Some(other_id) = params.get("other_id").and_then(|v| v.as_str()) {
            let Some(other) = primal.get_tensor(other_id) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Tensor not found: {other_id}"),
                );
            };
            tensor.add(&other)
        } else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                "Provide either 'other_id' or 'scalar'",
            );
        };
        match result {
            Ok(out) => tensor_result_response(primal, out, id),
            Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.add failed: {e}")),
        }
    } else if let Some(cpu_t) = primal.get_cpu_tensor(tid) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
        )]
        let new_data = if let Some(scalar) = params.get("scalar").and_then(|v| v.as_f64()) {
            cpu_t.data.iter().map(|&v| v + scalar as f32).collect()
        } else if let Some(other_id) = params.get("other_id").and_then(|v| v.as_str()) {
            let Some(other) = primal.get_cpu_tensor(other_id) else {
                return JsonRpcResponse::error(
                    id,
                    INVALID_PARAMS,
                    format!("Tensor not found: {other_id}"),
                );
            };
            cpu_t
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(&a, &b)| a + b)
                .collect()
        } else {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                "Provide either 'other_id' or 'scalar'",
            );
        };
        cpu_tensor_result(primal, new_data, cpu_t.shape, id)
    } else {
        JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"))
    }
}

/// `tensor.scale` — multiply tensor by scalar. CPU fallback on headless hosts.
pub(super) async fn tensor_scale(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };
    let Some(scalar) = params.get("scalar").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: scalar");
    };

    if let Some(tensor) = primal.get_tensor(tid) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
        )]
        match tensor.mul_scalar(scalar as f32) {
            Ok(out) => tensor_result_response(primal, out, id),
            Err(e) => {
                JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.scale failed: {e}"))
            }
        }
    } else if let Some(cpu_t) = primal.get_cpu_tensor(tid) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
        )]
        let new_data = cpu_t.data.iter().map(|&v| v * scalar as f32).collect();
        cpu_tensor_result(primal, new_data, cpu_t.shape, id)
    } else {
        JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"))
    }
}

/// `tensor.clamp` — clamp tensor values to [min, max]. CPU fallback on headless hosts.
pub(super) async fn tensor_clamp(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };
    let Some(min_val) = params.get("min").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: min");
    };
    let Some(max_val) = params.get("max").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: max");
    };

    if let Some(tensor) = primal.get_tensor(tid) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
        )]
        match (*tensor).clone().clamp_wgsl(min_val as f32, max_val as f32) {
            Ok(out) => tensor_result_response(primal, out, id),
            Err(e) => {
                JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.clamp failed: {e}"))
            }
        }
    } else if let Some(cpu_t) = primal.get_cpu_tensor(tid) {
        #[expect(
            clippy::cast_possible_truncation,
            reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
        )]
        let (lo, hi) = (min_val as f32, max_val as f32);
        let new_data = cpu_t.data.iter().map(|&v| v.clamp(lo, hi)).collect();
        cpu_tensor_result(primal, new_data, cpu_t.shape, id)
    } else {
        JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"))
    }
}

/// `tensor.reduce` — reduce tensor to scalar (sum, mean, max, min).
/// Works on both GPU and CPU tensors.
pub(super) async fn tensor_reduce(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };
    let op = params.get("op").and_then(|v| v.as_str()).unwrap_or("sum");

    let data: Vec<f32> = if let Some(tensor) = primal.get_tensor(tid) {
        match tensor.to_vec() {
            Ok(d) => d,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("Failed to read tensor: {e}"),
                );
            }
        }
    } else if let Some(cpu_t) = primal.get_cpu_tensor(tid) {
        cpu_t.data
    } else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"));
    };

    #[expect(
        clippy::cast_precision_loss,
        reason = "tensor element count fits f64 mantissa for any practical tensor"
    )]
    let result: f64 = match op {
        "sum" => data.iter().map(|&v| f64::from(v)).sum(),
        "mean" => {
            if data.is_empty() {
                0.0
            } else {
                data.iter().map(|&v| f64::from(v)).sum::<f64>() / data.len() as f64
            }
        }
        "max" => data
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max),
        "min" => data
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::INFINITY, f64::min),
        _ => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Unknown reduce op: {op} (use sum, mean, max, min)"),
            );
        }
    };

    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": result, "status": "completed", "value": result, "op": op }),
    )
}

/// `tensor.sigmoid` — element-wise sigmoid. CPU fallback on headless hosts.
pub(super) async fn tensor_sigmoid(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };

    if let Some(tensor) = primal.get_tensor(tid) {
        match (*tensor).clone().sigmoid() {
            Ok(out) => tensor_result_response(primal, out, id),
            Err(e) => {
                JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.sigmoid failed: {e}"))
            }
        }
    } else if let Some(cpu_t) = primal.get_cpu_tensor(tid) {
        let new_data = cpu_t
            .data
            .iter()
            .map(|&v| 1.0 / (1.0 + (-v).exp()))
            .collect();
        cpu_tensor_result(primal, new_data, cpu_t.shape, id)
    } else {
        JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"))
    }
}

fn tensor_result_response(
    primal: &BarraCudaPrimal,
    tensor: barracuda::tensor::Tensor,
    id: Value,
) -> JsonRpcResponse {
    let shape: Vec<usize> = tensor.shape().to_vec();
    let elements = shape.iter().product::<usize>();
    let result_id = primal.store_tensor(tensor);
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

/// CPU matmul for two CPU-resident tensors (2D matrices).
fn cpu_matmul(
    primal: &BarraCudaPrimal,
    lhs: &CpuTensor,
    rhs: &CpuTensor,
    id: Value,
) -> JsonRpcResponse {
    if lhs.shape.len() != 2 || rhs.shape.len() != 2 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "CPU matmul requires 2D tensors");
    }
    let (m, k) = (lhs.shape[0], lhs.shape[1]);
    let (k2, n) = (rhs.shape[0], rhs.shape[1]);
    if k != k2 {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Shape mismatch: lhs [{m}x{k}], rhs [{k2}x{n}]"),
        );
    }
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum = lhs.data[i * k + p].mul_add(rhs.data[p * n + j], sum);
            }
            out[i * n + j] = sum;
        }
    }
    cpu_tensor_result(primal, out, vec![m, n], id)
}

/// Store a CPU result tensor and return a standard tensor response.
fn cpu_tensor_result(
    primal: &BarraCudaPrimal,
    data: Vec<f32>,
    shape: Vec<usize>,
    id: Value,
) -> JsonRpcResponse {
    let elements = shape.iter().product::<usize>();
    let result_id = primal.store_cpu_tensor(CpuTensor {
        data,
        shape: shape.clone(),
    });
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

/// `tensor.matmul_inline` — matrix multiply with inline data (no handle round-trip).
///
/// Springs can send `lhs` and `rhs` as nested arrays directly and receive
/// the product matrix in the response, avoiding the create→operate→extract
/// three-call pattern for simple cases.
pub(super) fn tensor_matmul_inline(params: &Value, id: Value) -> JsonRpcResponse {
    let (Some(lhs_rows), Some(rhs_rows)) = (
        params.get("lhs").and_then(|v| v.as_array()).map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    r.as_array()
                        .map(|cols| cols.iter().filter_map(|c| c.as_f64()).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>()
        }),
        params.get("rhs").and_then(|v| v.as_array()).map(|rows| {
            rows.iter()
                .filter_map(|r| {
                    r.as_array()
                        .map(|cols| cols.iter().filter_map(|c| c.as_f64()).collect::<Vec<_>>())
                })
                .collect::<Vec<_>>()
        }),
    ) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required params: lhs and rhs (2D arrays)",
        );
    };
    let m = lhs_rows.len();
    if m == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "lhs must be non-empty");
    }
    let k = lhs_rows[0].len();
    if k == 0 || lhs_rows.iter().any(|r| r.len() != k) {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "lhs rows must have consistent length");
    }
    let n = if rhs_rows.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "rhs must be non-empty");
    } else {
        rhs_rows[0].len()
    };
    if rhs_rows.len() != k || rhs_rows.iter().any(|r| r.len() != n) {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("Shape mismatch: lhs is [{m}x{k}], rhs must be [{k}x{n}]"),
        );
    }
    let mut result = vec![vec![0.0; n]; m];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum = lhs_rows[i][p].mul_add(rhs_rows[p][j], sum);
            }
            result[i][j] = sum;
        }
    }
    JsonRpcResponse::success(id, serde_json::json!({ "result": result, "shape": [m, n] }))
}
