// SPDX-License-Identifier: AGPL-3.0-or-later
//! Named compute dispatch and shape parsing helpers.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// Parse a JSON array of u64 values into a `Vec<usize>`, returning `None` if
/// any dimension overflows the platform `usize`.
pub(super) fn parse_shape(arr: &[Value]) -> Option<Vec<usize>> {
    arr.iter()
        .filter_map(|v| v.as_u64())
        .map(usize::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()
        .ok()
}

/// `barracuda.compute.dispatch` — Dispatch a named compute operation.
///
/// Rather than accepting raw WGSL (which would require shader security auditing),
/// this dispatches named operations from barraCuda's shader library. Pass input
/// data and the operation produces output stored in the tensor store.
pub(super) async fn compute_dispatch(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
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
