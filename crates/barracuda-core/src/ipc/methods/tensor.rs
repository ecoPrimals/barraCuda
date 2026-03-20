// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tensor creation and matmul handlers.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::compute::parse_shape;
use crate::BarraCudaPrimal;
use serde_json::Value;

/// `barracuda.tensor.create` — Create a real tensor on the GPU device.
pub(super) async fn tensor_create(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
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
pub(super) async fn tensor_matmul(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
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
