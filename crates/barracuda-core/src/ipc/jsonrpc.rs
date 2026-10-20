// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC 2.0 protocol — re-exports canonical types from cellmembrane-types
//! with barraCuda-specific extensions.

// ── Canonical types (shared across all organisms) ──────────────────────
pub use cellmembrane_types::ecosystem::{
    JsonRpcError, JsonRpcRequest, JsonRpcResponse,
};

use serde_json::Value;

// ── Standard error codes (re-exported from canonical) ──────────────────
/// JSON-RPC 2.0 error code: parse error.
pub const PARSE_ERROR: i32 = JsonRpcError::PARSE_ERROR;
/// JSON-RPC 2.0 error code: invalid request.
pub const INVALID_REQUEST: i32 = JsonRpcError::INVALID_REQUEST;
/// JSON-RPC 2.0 error code: method not found.
pub const METHOD_NOT_FOUND: i32 = JsonRpcError::METHOD_NOT_FOUND;
/// JSON-RPC 2.0 error code: invalid params.
pub const INVALID_PARAMS: i32 = JsonRpcError::INVALID_PARAMS;
/// JSON-RPC 2.0 error code: internal error.
pub const INTERNAL_ERROR: i32 = JsonRpcError::INTERNAL_ERROR;

// ── barraCuda extensions ──────────────────────────────────────────────

/// Extension trait for barraCuda-specific request handling.
pub trait BarracudaRequest {
    /// Extract the request ID, defaulting to `null` for notifications.
    fn id_or_null(&self) -> Value;

    /// Validate this is a proper JSON-RPC 2.0 request.
    fn validate(&self) -> Result<(), JsonRpcResponse>;
}

impl BarracudaRequest for JsonRpcRequest {
    fn id_or_null(&self) -> Value {
        self.id.clone().unwrap_or(Value::Null)
    }

    fn validate(&self) -> Result<(), JsonRpcResponse> {
        if self.jsonrpc != "2.0" {
            return Err(JsonRpcResponse::error(
                self.id_or_null(),
                INVALID_REQUEST,
                "jsonrpc field must be \"2.0\"",
            ));
        }
        if self.method.is_empty() {
            return Err(JsonRpcResponse::error(
                self.id_or_null(),
                INVALID_REQUEST,
                "method must not be empty",
            ));
        }
        Ok(())
    }
}

/// Extension: create a parse error response (no valid ID available).
pub fn parse_error_response() -> JsonRpcResponse {
    JsonRpcResponse::error(Value::Null, PARSE_ERROR, "Parse error")
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_request() {
        let json = r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":1}"#;
        let req: JsonRpcRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.method, "device.list");
        assert_eq!(req.id, Some(Value::Number(1.into())));
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_success_response() {
        let resp = JsonRpcResponse::success(
            Value::Number(1.into()),
            serde_json::json!({"devices": []}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_error_response() {
        let resp = JsonRpcResponse::error(
            Value::Number(1.into()),
            METHOD_NOT_FOUND,
            "Method not found",
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"error\""));
        assert!(json.contains("-32601"));
    }

    #[test]
    fn test_invalid_version() {
        let req = JsonRpcRequest {
            jsonrpc: "1.0".to_string(),
            method: "test".to_string(),
            params: None,
            id: Some(Value::Number(1.into())),
        };
        assert!(req.validate().is_err());
    }
}
