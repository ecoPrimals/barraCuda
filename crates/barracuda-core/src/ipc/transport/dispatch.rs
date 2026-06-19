// SPDX-License-Identifier: AGPL-3.0-or-later
use super::super::jsonrpc::{JsonRpcRequest, JsonRpcResponse};
use super::super::methods;
use crate::BarraCudaPrimal;

pub(super) const SERIALIZATION_ERROR: &str =
    r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Serialization error"},"id":null}"#;

/// Validate and dispatch a parsed `JsonRpcRequest`, moving the `id` to avoid
/// cloning the `serde_json::Value` on every request.
///
/// Returns `None` for notifications (requests without `id`), per JSON-RPC 2.0
/// spec: "The Server MUST NOT reply to a Notification".
pub(super) async fn handle_request(
    primal: &BarraCudaPrimal,
    request: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    if let Err(err_resp) = request.validate() {
        return Some(err_resp);
    }

    let JsonRpcRequest {
        id, method, params, ..
    } = request;
    let id = match id {
        Some(v) if !v.is_null() => v,
        _ => return None,
    };
    Some(methods::dispatch(primal, &method, &params, id).await)
}

/// Parse a single line of JSON-RPC and dispatch to the method handler.
pub(super) async fn handle_line(primal: &BarraCudaPrimal, line: &str) -> Option<JsonRpcResponse> {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(_) => return Some(JsonRpcResponse::parse_error()),
    };
    handle_request(primal, request).await
}

/// Handle a JSON-RPC 2.0 batch request (JSON array on a single line).
///
/// Per JSON-RPC 2.0 spec §6:
/// - An empty array is an invalid request.
/// - Each element is processed independently; notifications produce no entry.
/// - If all elements are notifications, no response is returned (`None`).
/// - Non-array elements within the batch produce individual parse errors.
///
/// Uses `serde_json::from_value` to avoid stringify/re-parse per element.
pub(super) async fn handle_batch(primal: &BarraCudaPrimal, line: &str) -> Option<String> {
    let items: Vec<serde_json::Value> = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            return Some(
                serde_json::to_string(&JsonRpcResponse::parse_error())
                    .unwrap_or_else(|_| SERIALIZATION_ERROR.to_string()),
            );
        }
    };

    if items.is_empty() {
        return Some(
            serde_json::to_string(&JsonRpcResponse::error(
                serde_json::Value::Null,
                super::super::jsonrpc::INVALID_REQUEST,
                "empty batch",
            ))
            .unwrap_or_else(|_| SERIALIZATION_ERROR.to_string()),
        );
    }

    let mut responses = Vec::new();
    for item in items {
        match serde_json::from_value::<JsonRpcRequest>(item) {
            Ok(request) => {
                if let Some(resp) = handle_request(primal, request).await {
                    responses.push(resp);
                }
            }
            Err(_) => responses.push(JsonRpcResponse::parse_error()),
        }
    }

    if responses.is_empty() {
        return None;
    }

    Some(serde_json::to_string(&responses).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string()))
}
