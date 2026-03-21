// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC 2.0 protocol implementation.
//!
//! Self-implemented per wateringHole standard: each primal owns its protocol
//! layer. No shared IPC crate, no external JSON-RPC framework.

use serde::{Deserialize, Serialize};

/// JSON-RPC 2.0 request.
#[derive(Debug, Clone, Deserialize)]
pub struct JsonRpcRequest {
    /// Protocol version — must be "2.0".
    pub jsonrpc: String,
    /// Method name in `{domain}.{operation}` format.
    pub method: String,
    /// Parameters (positional or named).
    #[serde(default)]
    pub params: serde_json::Value,
    /// Request ID (null for notifications).
    pub id: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 response.
#[derive(Debug, Clone, Serialize)]
pub struct JsonRpcResponse {
    /// Protocol version — always "2.0".
    pub jsonrpc: String,
    /// Result on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    /// Error on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
    /// Matching request ID.
    pub id: serde_json::Value,
}

/// JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    /// Error code.
    pub code: i32,
    /// Human-readable message.
    pub message: String,
    /// Additional data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<serde_json::Value>,
}

/// JSON-RPC 2.0 error code: parse error.
pub const PARSE_ERROR: i32 = -32700;
/// JSON-RPC 2.0 error code: invalid request.
pub const INVALID_REQUEST: i32 = -32600;
/// JSON-RPC 2.0 error code: method not found.
pub const METHOD_NOT_FOUND: i32 = -32601;
/// JSON-RPC 2.0 error code: invalid params.
pub const INVALID_PARAMS: i32 = -32602;
/// JSON-RPC 2.0 error code: internal error.
pub const INTERNAL_ERROR: i32 = -32603;

/// JSON-RPC 2.0 protocol version string.
const JSONRPC_VERSION: &str = "2.0";

impl JsonRpcResponse {
    /// Create a success response.
    pub fn success(id: serde_json::Value, result: serde_json::Value) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            result: Some(result),
            error: None,
            id,
        }
    }

    /// Create an error response.
    pub fn error(id: serde_json::Value, code: i32, message: impl Into<String>) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION.to_owned(),
            result: None,
            error: Some(JsonRpcError {
                code,
                message: message.into(),
                data: None,
            }),
            id,
        }
    }

    /// Create a parse error (no valid ID available).
    pub fn parse_error() -> Self {
        Self::error(serde_json::Value::Null, PARSE_ERROR, "Parse error")
    }
}

impl JsonRpcRequest {
    /// Extract the request ID, defaulting to `null` for notifications.
    fn id_or_null(&self) -> serde_json::Value {
        self.id.clone().unwrap_or(serde_json::Value::Null)
    }

    /// Validate this is a proper JSON-RPC 2.0 request.
    pub fn validate(&self) -> Result<(), JsonRpcResponse> {
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
        assert_eq!(req.id, Some(serde_json::Value::Number(1.into())));
        assert!(req.validate().is_ok());
    }

    #[test]
    fn test_success_response() {
        let resp = JsonRpcResponse::success(
            serde_json::Value::Number(1.into()),
            serde_json::json!({"devices": []}),
        );
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"result\""));
        assert!(!json.contains("\"error\""));
    }

    #[test]
    fn test_error_response() {
        let resp = JsonRpcResponse::error(
            serde_json::Value::Number(1.into()),
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
            params: serde_json::Value::Null,
            id: Some(serde_json::Value::Number(1.into())),
        };
        assert!(req.validate().is_err());
    }
}
