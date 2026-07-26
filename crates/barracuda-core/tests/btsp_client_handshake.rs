// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for the BTSP client-side handshake.
//!
//! Tests the `BtspClient::handshake()` flow against a mock security provider
//! and a mock BTSP-enforced target server — both spawned locally on UDS.
#![cfg(unix)]

use barracuda_core::ipc::btsp_client::BtspClient;
use barracuda_core::ipc::transport::TransportEndpoint;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::UnixListener;

/// Spawn a mock security provider that handles `btsp.session.create` and
/// `btsp.session.verify` RPCs — the minimal surface needed by `BtspClient`.
async fn spawn_mock_provider(path: &std::path::Path) -> tokio::task::JoinHandle<()> {
    if path.exists() {
        std::fs::remove_file(path).ok();
    }
    let listener = UnixListener::bind(path).expect("bind mock provider");

    tokio::spawn(async move {
        // Accept multiple connections (create and verify come on separate connects
        // per BtspClient's current design).
        loop {
            let Ok((stream, _)) = listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut reader = BufReader::new(stream);
                let mut line = String::new();
                if reader.read_line(&mut line).await.is_err() {
                    return;
                }
                let req: serde_json::Value = match serde_json::from_str(&line) {
                    Ok(v) => v,
                    Err(_) => return,
                };

                let method = req["method"].as_str().unwrap_or("");
                let response = match method {
                    "btsp.session.create" => json!({
                        "jsonrpc": "2.0",
                        "result": {
                            "client_ephemeral_pub": "Y2xpZW50X2VwaGVtZXJhbF9wdWI=",
                            "session_id": "mock-session-42"
                        },
                        "id": req["id"]
                    }),
                    "btsp.session.verify" => json!({
                        "jsonrpc": "2.0",
                        "result": {
                            "client_response": "bW9ja19obWFjX3Jlc3BvbnNl",
                            "verified": true
                        },
                        "id": req["id"]
                    }),
                    _ => json!({
                        "jsonrpc": "2.0",
                        "error": {"code": -32601, "message": "Method not found"},
                        "id": req["id"]
                    }),
                };

                let mut resp_line = serde_json::to_string(&response).unwrap();
                resp_line.push('\n');
                let _ = reader.get_mut().write_all(resp_line.as_bytes()).await;
            });
        }
    })
}

/// Spawn a mock BTSP-enforced target server that expects the ClientHello
/// protocol and responds with ServerHello + HandshakeComplete.
async fn spawn_mock_target(path: &std::path::Path) -> tokio::task::JoinHandle<()> {
    if path.exists() {
        std::fs::remove_file(path).ok();
    }
    let listener = UnixListener::bind(path).expect("bind mock target");

    tokio::spawn(async move {
        let Ok((stream, _)) = listener.accept().await else {
            return;
        };
        let mut reader = BufReader::new(stream);

        // Read ClientHello.
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let hello: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(hello["protocol"], "btsp");
        assert_eq!(hello["version"], 1);
        assert!(hello["client_ephemeral_pub"].is_string());

        // Send ServerHello.
        let server_hello = json!({
            "type": "ServerHello",
            "server_ephemeral_pub": "c2VydmVyX2VwaGVtZXJhbF9wdWI=",
            "challenge": "dGVzdF9jaGFsbGVuZ2U=",
            "session_id": "mock-session-42"
        });
        let mut hello_line = serde_json::to_string(&server_hello).unwrap();
        hello_line.push('\n');
        reader
            .get_mut()
            .write_all(hello_line.as_bytes())
            .await
            .unwrap();

        // Read ChallengeResponse.
        let mut cr_line = String::new();
        reader.read_line(&mut cr_line).await.unwrap();
        let cr: serde_json::Value = serde_json::from_str(&cr_line).unwrap();
        assert_eq!(cr["type"], "ChallengeResponse");
        assert!(cr["response"].is_string());

        // Send HandshakeComplete.
        let complete = json!({
            "type": "HandshakeComplete",
            "cipher": "null",
            "session_id": "mock-session-42"
        });
        let mut complete_line = serde_json::to_string(&complete).unwrap();
        complete_line.push('\n');
        reader
            .get_mut()
            .write_all(complete_line.as_bytes())
            .await
            .unwrap();
    })
}

#[tokio::test]
async fn btsp_client_full_handshake_flow() {
    let tmp = std::env::temp_dir();
    let provider_path = tmp.join(format!("btsp_test_provider_{}.sock", std::process::id()));
    let target_path = tmp.join(format!("btsp_test_target_{}.sock", std::process::id()));

    let _provider_handle = spawn_mock_provider(&provider_path).await;
    let _target_handle = spawn_mock_target(&target_path).await;

    // Brief yield to let listeners bind.
    tokio::task::yield_now().await;

    let client = BtspClient::new(TransportEndpoint::uds(
        provider_path.to_string_lossy().into_owned(),
    ));
    let target_endpoint = TransportEndpoint::uds(target_path.to_string_lossy().into_owned());

    let result = client.handshake(&target_endpoint, "null").await;

    // Cleanup sockets.
    std::fs::remove_file(&provider_path).ok();
    std::fs::remove_file(&target_path).ok();

    let result = result.expect("handshake should succeed");
    assert_eq!(result.session.session_id, "mock-session-42");
    assert_eq!(result.session.cipher.wire_name(), "null");
    assert!(!result.session.cipher.requires_key());
}

#[tokio::test]
async fn btsp_client_rejects_server_error() {
    let tmp = std::env::temp_dir();
    let provider_path = tmp.join(format!(
        "btsp_test_provider_err_{}.sock",
        std::process::id()
    ));
    let target_path = tmp.join(format!("btsp_test_target_err_{}.sock", std::process::id()));

    let _provider_handle = spawn_mock_provider(&provider_path).await;

    // Target that immediately sends an error.
    if target_path.exists() {
        std::fs::remove_file(&target_path).ok();
    }
    let target_listener = UnixListener::bind(&target_path).unwrap();
    tokio::spawn(async move {
        let Ok((stream, _)) = target_listener.accept().await else {
            return;
        };
        let mut reader = BufReader::new(stream);
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();

        let error_resp = json!({
            "error": {"reason": "strict mode: FAMILY_ID mismatch"}
        });
        let mut resp = serde_json::to_string(&error_resp).unwrap();
        resp.push('\n');
        reader.get_mut().write_all(resp.as_bytes()).await.unwrap();
    });

    tokio::task::yield_now().await;

    let client = BtspClient::new(TransportEndpoint::uds(
        provider_path.to_string_lossy().into_owned(),
    ));
    let target_endpoint = TransportEndpoint::uds(target_path.to_string_lossy().into_owned());

    let result = client.handshake(&target_endpoint, "null").await;

    std::fs::remove_file(&provider_path).ok();
    std::fs::remove_file(&target_path).ok();

    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("rejected"),
        "expected rejection error, got: {err_msg}"
    );
}

#[tokio::test]
async fn btsp_client_fails_when_provider_unreachable() {
    let target_endpoint = TransportEndpoint::uds("/tmp/nonexistent_btsp_target_xyzzy.sock");
    let provider_endpoint = TransportEndpoint::uds("/tmp/nonexistent_btsp_provider_xyzzy.sock");
    let client = BtspClient::new(provider_endpoint);

    let result = client.handshake(&target_endpoint, "null").await;
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("security provider"),
        "expected provider connection error, got: {err_msg}"
    );
}
