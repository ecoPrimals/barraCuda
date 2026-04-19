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
use crate::BarraCudaPrimal;
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

    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

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

/// `tensor.add` — element-wise add of two tensors or tensor + scalar.
///
/// Accepts `{"tensor_id": "...", "other_id": "..."}` for tensor-tensor add,
/// or `{"tensor_id": "...", "scalar": 1.0}` for broadcast scalar add.
pub(super) async fn tensor_add(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };
    let Some(tensor) = primal.get_tensor(tid) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"));
    };

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
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Provide either 'other_id' or 'scalar'");
    };

    match result {
        Ok(out) => tensor_result_response(primal, out, id),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.add failed: {e}")),
    }
}

/// `tensor.scale` — multiply tensor by scalar.
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
    let Some(tensor) = primal.get_tensor(tid) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"));
    };

    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
    )]
    match tensor.mul_scalar(scalar as f32) {
        Ok(out) => tensor_result_response(primal, out, id),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.scale failed: {e}")),
    }
}

/// `tensor.clamp` — clamp tensor values to [min, max] via GPU WGSL.
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
    let Some(tensor) = primal.get_tensor(tid) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"));
    };

    #[expect(
        clippy::cast_possible_truncation,
        reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
    )]
    match (*tensor).clone().clamp_wgsl(min_val as f32, max_val as f32) {
        Ok(out) => tensor_result_response(primal, out, id),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.clamp failed: {e}")),
    }
}

/// `tensor.reduce` — reduce tensor to scalar (sum, mean, max, min).
pub(super) async fn tensor_reduce(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };
    let op = params.get("op").and_then(|v| v.as_str()).unwrap_or("sum");
    let Some(tensor) = primal.get_tensor(tid) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"));
    };

    let data = match tensor.to_vec() {
        Ok(d) => d,
        Err(e) => {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("Failed to read tensor: {e}"),
            );
        }
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

/// `tensor.sigmoid` — element-wise sigmoid via GPU WGSL.
pub(super) async fn tensor_sigmoid(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(tid) = params.get("tensor_id").and_then(|v| v.as_str()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: tensor_id");
    };
    let Some(tensor) = primal.get_tensor(tid) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, format!("Tensor not found: {tid}"));
    };

    match (*tensor).clone().sigmoid() {
        Ok(out) => tensor_result_response(primal, out, id),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("tensor.sigmoid failed: {e}")),
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
