// SPDX-License-Identifier: AGPL-3.0-or-later
//! Low-level JSON-RPC 2.0 transport over TCP.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Low-level JSON-RPC 2.0 call over TCP.
///
/// Opens a fresh TCP connection per call (simple, stateless). For the
/// shader compilation use case, connection overhead is negligible compared
/// to compilation time.
pub async fn jsonrpc_call<P: Serialize, R: for<'de> Deserialize<'de>>(
    addr: &str,
    method: &str,
    params: &P,
) -> crate::error::Result<R> {
    use crate::error::BarracudaError;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": [params],
        "id": 1,
    });
    let body = serde_json::to_string(&request)
        .map_err(|e| BarracudaError::Internal(format!("JSON-RPC serialize: {e}")))?;

    let host_port = addr.trim_start_matches("http://");
    let http_request = format!(
        "POST / HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = TcpStream::connect(host_port)
        .await
        .map_err(|e| BarracudaError::Internal(format!("TCP connect to {host_port}: {e}")))?;

    stream
        .write_all(http_request.as_bytes())
        .await
        .map_err(|e| BarracudaError::Internal(format!("TCP write: {e}")))?;

    let mut response_buf = Vec::new();
    stream
        .read_to_end(&mut response_buf)
        .await
        .map_err(|e| BarracudaError::Internal(format!("TCP read: {e}")))?;

    let response_str = String::from_utf8_lossy(&response_buf);

    let json_start = response_str
        .find('{')
        .ok_or_else(|| BarracudaError::Internal("no JSON body in HTTP response".into()))?;
    let json_body = &response_str[json_start..];

    let mut rpc_response: serde_json::Value = serde_json::from_str(json_body)
        .map_err(|e| BarracudaError::Internal(format!("JSON parse: {e}")))?;

    if let Some(error) = rpc_response.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(BarracudaError::Internal(format!("JSON-RPC error: {msg}")));
    }

    let result = rpc_response
        .as_object_mut()
        .and_then(|obj| obj.remove("result"))
        .ok_or_else(|| BarracudaError::Internal("no result field in JSON-RPC response".into()))?;

    serde_json::from_value(result)
        .map_err(|e| BarracudaError::Internal(format!("deserialize result: {e}")))
}

/// Convert WGSL to SPIR-V words using naga (local, no IPC).
pub fn wgsl_to_spirv(wgsl: &str) -> Option<Vec<u32>> {
    let module = match naga::front::wgsl::parse_str(wgsl) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!("coralReef: WGSL parse failed: {e}");
            return None;
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = match validator.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            tracing::debug!("coralReef: WGSL validation failed: {e}");
            return None;
        }
    };

    let options = naga::back::spv::Options {
        lang_version: (1, 5),
        ..Default::default()
    };
    let pipeline_options = None;

    match naga::back::spv::write_vec(&module, &info, &options, pipeline_options) {
        Ok(words) => Some(words),
        Err(e) => {
            tracing::debug!("coralReef: SPIR-V emit failed: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wgsl_to_spirv_valid_shader() {
        let wgsl = r"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                output[gid.x] = input[gid.x] * 2.0;
            }
        ";
        let spirv = wgsl_to_spirv(wgsl);
        assert!(spirv.is_some(), "valid WGSL should produce SPIR-V");
        let words = spirv.unwrap();
        assert!(!words.is_empty());
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic number");
    }

    #[test]
    fn wgsl_to_spirv_invalid_shader() {
        let spirv = wgsl_to_spirv("this is not valid wgsl {{{");
        assert!(spirv.is_none());
    }

    #[test]
    fn wgsl_to_spirv_empty_produces_module() {
        let spirv = wgsl_to_spirv("");
        assert!(
            spirv.is_some(),
            "empty WGSL is a valid empty module in naga"
        );
    }
}
