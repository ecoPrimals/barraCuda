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
) -> Result<R, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": [params],
        "id": 1,
    });
    let body = serde_json::to_string(&request).map_err(|e| e.to_string())?;

    let host_port = addr.trim_start_matches("http://");
    let http_request = format!(
        "POST / HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = TcpStream::connect(host_port)
        .await
        .map_err(|e| format!("TCP connect to {host_port}: {e}"))?;

    stream
        .write_all(http_request.as_bytes())
        .await
        .map_err(|e| format!("TCP write: {e}"))?;

    let mut response_buf = Vec::new();
    stream
        .read_to_end(&mut response_buf)
        .await
        .map_err(|e| format!("TCP read: {e}"))?;

    let response_str = String::from_utf8_lossy(&response_buf);

    let json_start = response_str
        .find('{')
        .ok_or("no JSON body in HTTP response")?;
    let json_body = &response_str[json_start..];

    let rpc_response: serde_json::Value =
        serde_json::from_str(json_body).map_err(|e| format!("JSON parse: {e}"))?;

    if let Some(error) = rpc_response.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("JSON-RPC error: {msg}"));
    }

    let result = rpc_response
        .get("result")
        .ok_or("no result field in JSON-RPC response")?;

    serde_json::from_value(result.clone()).map_err(|e| format!("deserialize result: {e}"))
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
