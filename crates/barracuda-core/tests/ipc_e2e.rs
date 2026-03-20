// SPDX-License-Identifier: AGPL-3.0-or-later
#![allow(clippy::unwrap_used, clippy::single_match_else)]
//! End-to-end IPC tests for barraCuda JSON-RPC 2.0 over TCP.
//!
//! These tests start a real TCP server, connect as a client, send JSON-RPC
//! requests over the wire, and assert on the responses. This exercises the
//! full transport pipeline: serialization, framing, dispatch, and response.

#![expect(clippy::unwrap_used, reason = "E2E tests use unwrap for clarity")]

use serde_json::{Value, json};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// Start a JSON-RPC server on an ephemeral port, returning the bound address.
///
/// The server runs in a background task and is aborted when the returned
/// `JoinHandle` is dropped/aborted.
async fn start_server() -> (String, tokio::task::JoinHandle<()>) {
    let primal = Arc::new(barracuda_core::BarraCudaPrimal::new());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();

    let handle = tokio::spawn(async move {
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            let p = Arc::clone(&primal);
            tokio::spawn(async move {
                let (reader, mut writer) = stream.into_split();
                let mut lines = BufReader::new(reader).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let req: Value = match serde_json::from_str(&line) {
                        Ok(v) => v,
                        Err(_) => {
                            let err = json!({
                                "jsonrpc": "2.0",
                                "error": {"code": -32700, "message": "Parse error"},
                                "id": null
                            });
                            let mut s = serde_json::to_string(&err).unwrap();
                            s.push('\n');
                            let _ = writer.write_all(s.as_bytes()).await;
                            continue;
                        }
                    };

                    let method = req["method"].as_str().unwrap_or("");
                    let params = &req["params"];
                    let id = req["id"].clone();

                    let resp = barracuda_core::ipc::methods::dispatch(&p, method, params, id).await;
                    let mut s = serde_json::to_string(&resp).unwrap();
                    s.push('\n');
                    if writer.write_all(s.as_bytes()).await.is_err() {
                        break;
                    }
                }
            });
        }
    });

    (addr, handle)
}

/// Send a JSON-RPC request and read the response.
async fn rpc_call(stream: &mut TcpStream, req: Value) -> Value {
    let mut payload = serde_json::to_string(&req).unwrap();
    payload.push('\n');
    stream.write_all(payload.as_bytes()).await.unwrap();

    let mut reader = BufReader::new(&mut *stream);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    serde_json::from_str(&line).unwrap()
}

// --- E2E Tests ---

#[tokio::test]
async fn e2e_health_check() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.health.check",
            "params": {},
            "id": 1
        }),
    )
    .await;

    assert_eq!(resp["jsonrpc"], "2.0");
    assert_eq!(resp["id"], 1);
    assert!(resp["result"].is_object(), "expected result: {resp}");
    assert_eq!(resp["result"]["name"], "barraCuda");

    handle.abort();
}

#[tokio::test]
async fn e2e_device_list() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.device.list",
            "params": {},
            "id": 2
        }),
    )
    .await;

    assert_eq!(resp["id"], 2);
    assert!(resp["result"].is_object());
    assert!(resp["result"]["devices"].is_array());

    handle.abort();
}

#[tokio::test]
async fn e2e_tolerances_get() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.tolerances.get",
            "params": {"name": "fhe"},
            "id": 3
        }),
    )
    .await;

    assert_eq!(resp["id"], 3);
    let result = &resp["result"];
    assert_eq!(result["name"], "fhe");
    assert_eq!(result["abs_tol"], 0.0);
    assert_eq!(result["rel_tol"], 0.0);

    handle.abort();
}

#[tokio::test]
async fn e2e_unknown_method() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "nonexistent.method",
            "params": {},
            "id": 4
        }),
    )
    .await;

    assert_eq!(resp["id"], 4);
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32601);

    handle.abort();
}

#[tokio::test]
async fn e2e_malformed_json() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    stream.write_all(b"{{bad json\n").await.unwrap();

    let mut reader = BufReader::new(&mut stream);
    let mut line = String::new();
    reader.read_line(&mut line).await.unwrap();
    let resp: Value = serde_json::from_str(&line).unwrap();

    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32700);

    handle.abort();
}

#[tokio::test]
async fn e2e_multiple_requests_single_connection() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp1 = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.health.check",
            "params": {},
            "id": 10
        }),
    )
    .await;
    assert_eq!(resp1["id"], 10);
    assert!(resp1["result"].is_object());

    let resp2 = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.device.list",
            "params": {},
            "id": 11
        }),
    )
    .await;
    assert_eq!(resp2["id"], 11);
    assert!(resp2["result"]["devices"].is_array());

    let resp3 = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.tolerances.get",
            "params": {"name": "f32"},
            "id": 12
        }),
    )
    .await;
    assert_eq!(resp3["id"], 12);
    let tols = &resp3["result"];
    assert!(tols["abs_tol"].as_f64().unwrap() > 0.0);
    assert!(tols["rel_tol"].as_f64().unwrap() > 0.0);

    handle.abort();
}

#[tokio::test]
async fn e2e_validate_gpu_stack() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.validate.gpu_stack",
            "params": {},
            "id": 20
        }),
    )
    .await;

    assert_eq!(resp["id"], 20);
    // GPU may or may not be available; either result or error is acceptable
    assert!(resp["result"].is_object() || resp["error"].is_object());

    handle.abort();
}

#[tokio::test]
async fn e2e_device_probe() {
    let (addr, handle) = start_server().await;
    let mut stream = TcpStream::connect(&addr).await.unwrap();

    let resp = rpc_call(
        &mut stream,
        json!({
            "jsonrpc": "2.0",
            "method": "barracuda.device.probe",
            "params": {},
            "id": 30
        }),
    )
    .await;

    assert_eq!(resp["id"], 30);
    assert!(resp["result"].is_object());
    assert!(resp["result"]["available"].is_boolean());

    handle.abort();
}
