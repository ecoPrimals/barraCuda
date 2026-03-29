// SPDX-License-Identifier: AGPL-3.0-or-later
//! Low-level JSON-RPC 2.0 transport over TCP and Unix sockets.
//!
//! Supports two wire formats per wateringHole IPC v3.1:
//! - **Newline-delimited** (mandatory for inter-primal composition): one JSON
//!   object per line, `\n` delimiter. Tried first for both TCP and Unix socket.
//! - **HTTP-wrapped** (legacy fallback): standard `POST / HTTP/1.1` framing.
//!   Used when the newline-delimited attempt returns an HTTP response.

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// JSON-RPC 2.0 call over TCP, with newline-delimited v3.1 framing.
///
/// Opens a fresh TCP connection per call (simple, stateless). Tries
/// newline-delimited framing first (wateringHole v3.1 mandatory), falls
/// back to HTTP-wrapped framing for pre-v3.1 endpoints.
pub async fn jsonrpc_call<P: Serialize, R: for<'de> Deserialize<'de>>(
    addr: &str,
    method: &str,
    params: &P,
) -> crate::error::Result<R> {
    let host_port = addr.trim_start_matches("http://");

    if let Some(path) = host_port.strip_prefix("unix:") {
        return jsonrpc_call_unix::<P, R>(path, method, params).await;
    }

    match jsonrpc_call_ndjson_tcp::<P, R>(host_port, method, params).await {
        Ok(result) => Ok(result),
        Err(_) => jsonrpc_call_http::<P, R>(host_port, method, params).await,
    }
}

/// Newline-delimited JSON-RPC 2.0 over TCP (wateringHole v3.1).
async fn jsonrpc_call_ndjson_tcp<P: Serialize, R: for<'de> Deserialize<'de>>(
    host_port: &str,
    method: &str,
    params: &P,
) -> crate::error::Result<R> {
    use crate::error::BarracudaError;

    let stream = TcpStream::connect(host_port)
        .await
        .map_err(|e| BarracudaError::Internal(format!("TCP connect to {host_port}: {e}")))?;

    jsonrpc_call_ndjson_stream(stream, method, params).await
}

/// Newline-delimited JSON-RPC 2.0 over a Unix socket.
#[cfg(unix)]
async fn jsonrpc_call_unix<P: Serialize, R: for<'de> Deserialize<'de>>(
    path: &str,
    method: &str,
    params: &P,
) -> crate::error::Result<R> {
    use crate::error::BarracudaError;

    let stream = tokio::net::UnixStream::connect(path)
        .await
        .map_err(|e| BarracudaError::Internal(format!("Unix connect to {path}: {e}")))?;

    jsonrpc_call_ndjson_stream(stream, method, params).await
}

#[cfg(not(unix))]
async fn jsonrpc_call_unix<P: Serialize, R: for<'de> Deserialize<'de>>(
    path: &str,
    _method: &str,
    _params: &P,
) -> crate::error::Result<R> {
    Err(crate::error::BarracudaError::Internal(format!(
        "Unix sockets not supported on this platform: {path}"
    )))
}

/// Send a JSON-RPC request over a newline-delimited stream (TCP or Unix).
async fn jsonrpc_call_ndjson_stream<
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
    P: Serialize,
    R: for<'de> Deserialize<'de>,
>(
    mut stream: S,
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
    let mut body = serde_json::to_string(&request)
        .map_err(|e| BarracudaError::Internal(format!("JSON-RPC serialize: {e}")))?;
    body.push('\n');

    stream
        .write_all(body.as_bytes())
        .await
        .map_err(|e| BarracudaError::Internal(format!("ndjson write: {e}")))?;
    stream
        .flush()
        .await
        .map_err(|e| BarracudaError::Internal(format!("ndjson flush: {e}")))?;

    let mut reader = BufReader::new(&mut stream);
    let mut line = String::new();
    reader
        .read_line(&mut line)
        .await
        .map_err(|e| BarracudaError::Internal(format!("ndjson read: {e}")))?;

    if line.is_empty() {
        return Err(BarracudaError::Internal(
            "empty response from ndjson endpoint".into(),
        ));
    }

    parse_jsonrpc_response(&line)
}

/// HTTP-wrapped JSON-RPC 2.0 over TCP (legacy fallback for pre-v3.1 endpoints).
async fn jsonrpc_call_http<P: Serialize, R: for<'de> Deserialize<'de>>(
    host_port: &str,
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

    parse_jsonrpc_response(json_body)
}

/// Parse a JSON-RPC 2.0 response string, extracting the result or error.
fn parse_jsonrpc_response<R: for<'de> Deserialize<'de>>(
    json_body: &str,
) -> crate::error::Result<R> {
    use crate::error::BarracudaError;

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

    #[test]
    fn parse_jsonrpc_response_ok() {
        let json = r#"{"jsonrpc":"2.0","result":"hello","id":1}"#;
        let result: crate::error::Result<String> = parse_jsonrpc_response(json);
        assert_eq!(result.unwrap(), "hello");
    }

    #[test]
    fn parse_jsonrpc_response_error() {
        let json =
            r#"{"jsonrpc":"2.0","error":{"code":-32600,"message":"Invalid Request"},"id":1}"#;
        let result: crate::error::Result<String> = parse_jsonrpc_response(json);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("Invalid Request"), "got: {err}");
    }

    #[test]
    fn parse_jsonrpc_response_no_result() {
        let json = r#"{"jsonrpc":"2.0","id":1}"#;
        let result: crate::error::Result<String> = parse_jsonrpc_response(json);
        assert!(result.is_err());
    }

    #[test]
    fn parse_jsonrpc_response_invalid_json() {
        let result: crate::error::Result<String> = parse_jsonrpc_response("not json");
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn ndjson_roundtrip_via_mock_server() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();

            let req: serde_json::Value = serde_json::from_str(&line).unwrap();
            assert_eq!(req["method"], "test.echo");

            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": "pong",
                "id": req["id"],
            });
            let mut resp_line = serde_json::to_string(&resp).unwrap();
            resp_line.push('\n');

            use tokio::io::AsyncWriteExt;
            stream.write_all(resp_line.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
        });

        let addr = format!("127.0.0.1:{port}");
        let result: String = jsonrpc_call(&addr, "test.echo", &"ping")
            .await
            .expect("ndjson call should succeed");
        assert_eq!(result, "pong");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn unix_socket_roundtrip_via_mock_server() {
        let dir = std::env::temp_dir().join("barracuda_test_unix_rpc");
        let _ = std::fs::create_dir_all(&dir);
        let sock_path = dir.join("test.sock");
        let _ = std::fs::remove_file(&sock_path);

        let sock_path_clone = sock_path.clone();
        let listener = tokio::net::UnixListener::bind(&sock_path_clone).unwrap();

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(&mut stream);
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();

            let req: serde_json::Value = serde_json::from_str(&line).unwrap();
            let resp = serde_json::json!({
                "jsonrpc": "2.0",
                "result": 42,
                "id": req["id"],
            });
            let mut resp_line = serde_json::to_string(&resp).unwrap();
            resp_line.push('\n');

            use tokio::io::AsyncWriteExt;
            stream.write_all(resp_line.as_bytes()).await.unwrap();
            stream.flush().await.unwrap();
        });

        let addr = format!("unix:{}", sock_path.display());
        let result: i64 = jsonrpc_call(&addr, "math.add", &[1, 2])
            .await
            .expect("unix socket call should succeed");
        assert_eq!(result, 42);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
