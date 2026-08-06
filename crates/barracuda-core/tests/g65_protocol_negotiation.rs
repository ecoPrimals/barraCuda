// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(unix)]
#![expect(clippy::unwrap_used, reason = "test assertions use unwrap for clarity")]
//! G65 Protocol Negotiation integration tests.
//!
//! Tests the full accept-loop integration: client connects to the barraCuda
//! IPC server, sends `PROTOCOLS: tarpc,jsonrpc\n`, receives `PROTOCOL: tarpc\n`,
//! and then makes a tarpc service call through the negotiated connection.
//!
//! Also tests backward compatibility: legacy clients that skip negotiation
//! default to JSON-RPC.

use std::sync::Arc;

/// Start a real IpcServer on a Unix domain socket, returning the socket path
/// and a handle to abort the server.
async fn start_g65_server(
    dir: &std::path::Path,
) -> (std::path::PathBuf, tokio::task::JoinHandle<()>) {
    let sock = dir.join("g65_test.sock");
    let primal = Arc::new(barracuda_core::BarraCudaPrimal::new());
    let server = barracuda_core::ipc::IpcServer::new(primal);

    let sock2 = sock.clone();
    let handle = tokio::spawn(async move {
        server
            .serve_unix(&sock2, None::<fn()>)
            .await
            .expect("serve_unix");
    });

    // Wait for the socket to appear
    for _ in 0..100 {
        if sock.exists() {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
    assert!(sock.exists(), "server socket did not appear");

    (sock, handle)
}

/// Helper: connect and negotiate G65 protocol.
async fn negotiate_protocol(
    sock: &std::path::Path,
    protocols: &str,
) -> (String, tokio::net::UnixStream) {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let mut stream = tokio::net::UnixStream::connect(sock)
        .await
        .expect("connect");

    stream
        .write_all(format!("PROTOCOLS: {protocols}\n").as_bytes())
        .await
        .expect("write PROTOCOLS");
    stream.flush().await.expect("flush");

    let mut reader = tokio::io::BufReader::new(&mut stream);
    let mut response = String::new();
    reader
        .read_line(&mut response)
        .await
        .expect("read response");

    let selected = response
        .trim()
        .strip_prefix("PROTOCOL: ")
        .expect("PROTOCOL: prefix")
        .to_string();

    (selected, stream)
}

#[tokio::test]
async fn g65_negotiate_jsonrpc() {
    let dir = tempfile::tempdir().unwrap();
    let (sock, handle) = start_g65_server(dir.path()).await;

    let (selected, mut stream) = negotiate_protocol(&sock, "jsonrpc").await;
    assert_eq!(selected, "jsonrpc");

    // After negotiation, send a JSON-RPC request
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "health.liveness",
        "id": 1
    });
    let mut line = serde_json::to_string(&request).unwrap();
    line.push('\n');
    stream.write_all(line.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    let mut buf_reader = tokio::io::BufReader::new(&mut stream);
    let mut response = String::new();
    buf_reader.read_line(&mut response).await.unwrap();

    let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(resp["result"]["status"], "alive");

    handle.abort();
}

#[cfg(feature = "tarpc-transport")]
#[tokio::test]
async fn g65_negotiate_tarpc_then_call() {
    use barracuda_core::rpc::BarraCudaServiceClient;
    use tokio_serde::formats::Json;

    let dir = tempfile::tempdir().unwrap();
    let (sock, handle) = start_g65_server(dir.path()).await;

    let (selected, stream) = negotiate_protocol(&sock, "tarpc,jsonrpc").await;
    assert_eq!(selected, "tarpc", "server should select tarpc when offered");

    // Wrap the negotiated stream in tarpc transport
    let transport = tarpc::serde_transport::new(
        tokio_util::codec::LengthDelimitedCodec::builder()
            .max_frame_length(256 * 1024 * 1024)
            .new_framed(stream),
        Json::default(),
    );

    let client = BarraCudaServiceClient::new(tarpc::client::Config::default(), transport).spawn();

    // Make a tarpc service call through the G65-negotiated connection
    let liveness = client
        .health_liveness(tarpc::context::current())
        .await
        .expect("tarpc call");
    assert_eq!(liveness.status, "alive");

    let info = client
        .identity_get(tarpc::context::current())
        .await
        .expect("identity call");
    assert_eq!(info.primal, "barracuda");
    assert_eq!(info.domain, "math");

    handle.abort();
}

#[tokio::test]
async fn g65_backward_compat_legacy_jsonrpc() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let dir = tempfile::tempdir().unwrap();
    let (sock, handle) = start_g65_server(dir.path()).await;

    // Legacy client: connect and send JSON-RPC directly (no negotiation header)
    let mut stream = tokio::net::UnixStream::connect(&sock)
        .await
        .expect("connect");

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "health.liveness",
        "id": 42
    });
    let mut line = serde_json::to_string(&request).unwrap();
    line.push('\n');
    stream.write_all(line.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    let mut buf_reader = tokio::io::BufReader::new(&mut stream);
    let mut response = String::new();
    buf_reader.read_line(&mut response).await.unwrap();

    let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(resp["result"]["status"], "alive");
    assert_eq!(resp["id"], 42);

    handle.abort();
}

#[tokio::test]
async fn g65_protocols_list_advertises_negotiation() {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let dir = tempfile::tempdir().unwrap();
    let (sock, handle) = start_g65_server(dir.path()).await;

    let mut stream = tokio::net::UnixStream::connect(&sock)
        .await
        .expect("connect");

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "protocols.list",
        "id": 1
    });
    let mut line = serde_json::to_string(&request).unwrap();
    line.push('\n');
    stream.write_all(line.as_bytes()).await.unwrap();
    stream.flush().await.unwrap();

    let mut buf_reader = tokio::io::BufReader::new(&mut stream);
    let mut response = String::new();
    buf_reader.read_line(&mut response).await.unwrap();

    let resp: serde_json::Value = serde_json::from_str(&response).unwrap();
    let result = &resp["result"];

    // Verify G65 negotiation is advertised
    assert_eq!(result["negotiation"]["g65"], true);
    let supported = result["negotiation"]["supported"]
        .as_array()
        .expect("supported array");
    assert!(supported.iter().any(|v| v.as_str() == Some("jsonrpc")));

    handle.abort();
}
