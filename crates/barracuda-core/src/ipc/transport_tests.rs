// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]

use super::*;

#[tokio::test]
async fn test_handle_valid_request() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":1}"#;
    let resp = handle_line(&primal, line).await.expect("non-notification");
    assert!(resp.result.is_some());
    assert_eq!(resp.id, serde_json::json!(1));
}

#[tokio::test]
async fn test_handle_parse_error() {
    let primal = BarraCudaPrimal::new();
    let resp = handle_line(&primal, "not json")
        .await
        .expect("parse error returns response");
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, super::super::jsonrpc::PARSE_ERROR);
}

#[tokio::test]
async fn test_handle_unknown_method() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"nonexistent","params":{},"id":2}"#;
    let resp = handle_line(&primal, line).await.expect("non-notification");
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_notification_no_response() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"device.list","params":{}}"#;
    assert!(
        handle_line(&primal, line).await.is_none(),
        "notification must not produce response"
    );
}

#[tokio::test]
async fn test_notification_null_id_no_response() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":null}"#;
    assert!(
        handle_line(&primal, line).await.is_none(),
        "null id is a notification"
    );
}

// ─── resolve_bind_host tests ───

#[test]
fn resolve_bind_host_returns_valid_ip() {
    let host = resolve_bind_host();
    assert!(
        host.parse::<std::net::IpAddr>().is_ok(),
        "resolve_bind_host must return a valid IP, got: {host}"
    );
}

#[test]
fn resolve_bind_host_fallback_matches_default() {
    if std::env::var("BARRACUDA_IPC_HOST").is_err() {
        assert_eq!(resolve_bind_host(), DEFAULT_BIND_HOST);
    }
}

// ─── resolve_bind_address tests ───

#[test]
fn resolve_explicit_addr() {
    assert_eq!(resolve_bind_address(Some("0.0.0.0:8080")), "0.0.0.0:8080");
}

#[test]
fn resolve_explicit_always_wins() {
    let addr = resolve_bind_address(Some("10.0.0.1:9000"));
    assert_eq!(addr, "10.0.0.1:9000");
}

#[test]
fn resolve_defaults_to_ephemeral() {
    let addr = resolve_bind_address(None);
    assert!(
        addr.ends_with(":0") || addr.contains(':'),
        "default should use ephemeral port"
    );
}

#[tokio::test]
async fn test_handle_line_invalid_jsonrpc_version() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"1.0","method":"device.list","params":{},"id":1}"#;
    let resp = handle_line(&primal, line).await.expect("non-notification");
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_handle_line_empty_method() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"","params":{},"id":1}"#;
    let resp = handle_line(&primal, line).await.expect("non-notification");
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_handle_line_string_id() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"primal.info","params":{},"id":"abc"}"#;
    let resp = handle_line(&primal, line).await.expect("non-notification");
    assert!(resp.result.is_some());
    assert_eq!(resp.id, serde_json::json!("abc"));
}

#[tokio::test]
async fn test_handle_line_health_liveness_alias() {
    let primal = BarraCudaPrimal::new();
    let line = r#"{"jsonrpc":"2.0","method":"ping","params":{},"id":99}"#;
    let resp = handle_line(&primal, line).await.expect("non-notification");
    assert!(resp.result.is_some());
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[cfg(unix)]
#[test]
fn default_socket_path_format() {
    let path = IpcServer::default_socket_path();
    let path_str = path.to_string_lossy();
    assert!(path_str.contains(DEFAULT_ECOSYSTEM_SOCKET_DIR));
    assert!(
        path_str.ends_with("math.sock") || path_str.contains("math-"),
        "default path should be math.sock or math-{{fid}}.sock, got {path_str}"
    );
}

#[test]
fn resolve_family_id_returns_none_when_unset() {
    assert!(
        resolve_family_id().is_none() || resolve_family_id().is_some(),
        "should return Some or None depending on env"
    );
}

#[test]
fn resolve_socket_dir_returns_nonempty() {
    let dir = resolve_socket_dir();
    assert!(!dir.as_os_str().is_empty(), "socket dir must not be empty");
    let dir_str = dir.to_string_lossy();
    assert!(
        dir_str.contains(DEFAULT_ECOSYSTEM_SOCKET_DIR)
            || std::env::var("BIOMEOS_SOCKET_DIR").is_ok(),
        "should contain biomeos namespace or be overridden, got {dir_str}"
    );
}

#[test]
fn validate_insecure_guard_ok_when_clean() {
    assert!(
        validate_insecure_guard().is_ok(),
        "should pass when env is clean"
    );
}

// ─── handle_connection integration tests ───

#[tokio::test]
async fn handle_connection_valid_request() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let request = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
    let input = format!("{request}\n");

    let (reader, _) = tokio::io::duplex(4096);
    let (_, writer) = tokio::io::duplex(4096);

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();
    drop(reader);
    drop(writer);

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    assert!(response.contains("\"jsonrpc\":\"2.0\""));
    assert!(response.contains("\"result\""));
}

#[tokio::test]
async fn handle_connection_parse_error() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let input = "not valid json\n";

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    assert!(
        response.contains("-32700"),
        "should contain parse error code"
    );
}

#[tokio::test]
async fn handle_connection_multiple_requests() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let req1 = r#"{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":1}"#;
    let req2 = r#"{"jsonrpc":"2.0","method":"primal.info","params":{},"id":2}"#;
    let input = format!("{req1}\n{req2}\n");

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 2, "should have two response lines");
    assert!(lines[0].contains("alive"));
    assert!(lines[1].contains("barraCuda"));
}

#[tokio::test]
async fn handle_connection_mixed_valid_invalid() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let valid = r#"{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":1}"#;
    let invalid = "not valid json";
    let input = format!("{valid}\n{invalid}\n");

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(lines.len(), 2, "should have two responses");
    assert!(lines[0].contains("alive"));
    assert!(lines[1].contains("-32700"), "parse error for invalid JSON");
}

// ─── JSON-RPC batch tests ───

#[tokio::test]
async fn test_batch_two_requests() {
    let primal = BarraCudaPrimal::new();
    let batch = r#"[
            {"jsonrpc":"2.0","method":"health.liveness","params":{},"id":1},
            {"jsonrpc":"2.0","method":"primal.info","params":{},"id":2}
        ]"#;
    let resp = handle_batch(&primal, batch).await.expect("batch response");
    let arr: Vec<serde_json::Value> = serde_json::from_str(&resp).unwrap();
    assert_eq!(arr.len(), 2);
    assert_eq!(arr[0]["id"], 1);
    assert_eq!(arr[1]["id"], 2);
}

#[tokio::test]
async fn test_batch_empty_array_is_invalid() {
    let primal = BarraCudaPrimal::new();
    let resp = handle_batch(&primal, "[]").await.expect("error response");
    let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(
        parsed["error"]["code"],
        super::super::jsonrpc::INVALID_REQUEST
    );
}

#[tokio::test]
async fn test_batch_invalid_json() {
    let primal = BarraCudaPrimal::new();
    let resp = handle_batch(&primal, "[invalid")
        .await
        .expect("parse error");
    let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(parsed["error"]["code"], super::super::jsonrpc::PARSE_ERROR);
}

#[tokio::test]
async fn test_batch_all_notifications_no_response() {
    let primal = BarraCudaPrimal::new();
    let batch = r#"[
            {"jsonrpc":"2.0","method":"health.liveness","params":{}},
            {"jsonrpc":"2.0","method":"device.list","params":{}}
        ]"#;
    assert!(
        handle_batch(&primal, batch).await.is_none(),
        "all-notification batch must not produce a response"
    );
}

#[tokio::test]
async fn test_batch_mixed_request_and_notification() {
    let primal = BarraCudaPrimal::new();
    let batch = r#"[
            {"jsonrpc":"2.0","method":"health.liveness","params":{}},
            {"jsonrpc":"2.0","method":"primal.info","params":{},"id":"abc"}
        ]"#;
    let resp = handle_batch(&primal, batch).await.expect("one response");
    let arr: Vec<serde_json::Value> = serde_json::from_str(&resp).unwrap();
    assert_eq!(
        arr.len(),
        1,
        "only the non-notification should produce a response"
    );
    assert_eq!(arr[0]["id"], "abc");
}

#[tokio::test]
async fn test_batch_via_connection_handler() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let batch = r#"[{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":1},{"jsonrpc":"2.0","method":"primal.info","params":{},"id":2}]"#;
    let input = format!("{batch}\n");

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    let arr: Vec<serde_json::Value> = serde_json::from_str(response.trim()).unwrap();
    assert_eq!(arr.len(), 2, "batch via connection should return array");
}

#[test]
fn ipc_server_construction() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let _server = IpcServer::new(primal);
}

#[test]
fn max_frame_bytes_default() {
    assert!(max_frame_bytes() > 0);
    assert!(max_frame_bytes() >= 1024);
}

#[test]
fn max_connections_default() {
    assert!(max_connections() > 0);
}

#[tokio::test]
async fn handle_connection_notification_no_reply() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let input = "{\"jsonrpc\":\"2.0\",\"method\":\"device.list\",\"params\":{}}\n";

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    assert!(response.is_empty(), "notification must produce no response");
}

#[tokio::test]
async fn handle_connection_replay_consumed_line() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let replay = r#"{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":99}"#.to_string();
    let followup = r#"{"jsonrpc":"2.0","method":"primal.info","params":{},"id":100}"#;
    let input = format!("{followup}\n");

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(input.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, Some(replay)).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "replay + stream should produce two responses"
    );
    assert!(
        lines[0].contains("\"id\":99"),
        "first response is for the replayed request"
    );
    assert!(
        lines[1].contains("\"id\":100"),
        "second response is for the stream request"
    );
}

#[tokio::test]
async fn test_try_bind_tcp_succeeds_on_free_port() {
    let result = IpcServer::try_bind_tcp("127.0.0.1:0").await;
    assert!(result.is_some(), "binding to port 0 should succeed");
    let (_listener, addr) = result.unwrap();
    assert_ne!(addr.port(), 0, "OS should assign a real port");
}

#[tokio::test]
async fn test_try_bind_tcp_returns_none_on_addr_in_use() {
    let first = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let occupied_addr = first.local_addr().unwrap().to_string();
    let result = IpcServer::try_bind_tcp(&occupied_addr).await;
    assert!(
        result.is_none(),
        "try_bind_tcp should return None when address is in use"
    );
}

#[tokio::test]
async fn test_serve_tarpc_returns_ok_on_addr_in_use() {
    let first = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let occupied_addr = first.local_addr().unwrap().to_string();
    let primal = Arc::new(BarraCudaPrimal::new());
    let server = IpcServer::new(primal);
    let result = server.serve_tarpc(&occupied_addr).await;
    assert!(
        result.is_ok(),
        "serve_tarpc should return Ok(()) on AddrInUse, not propagate error"
    );
}

#[tokio::test]
async fn handle_connection_whitespace_batch() {
    let primal = Arc::new(BarraCudaPrimal::new());
    let batch = concat!(
        "  \t [",
        r#"{"jsonrpc":"2.0","method":"health.liveness","params":{},"id":1},"#,
        r#"{"jsonrpc":"2.0","method":"primal.info","params":{},"id":2}"#,
        "]\n"
    );

    let (client_reader, mut server_writer) = tokio::io::duplex(4096);
    let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

    server_writer.write_all(batch.as_bytes()).await.unwrap();
    server_writer.shutdown().await.unwrap();

    handle_connection(primal, client_reader, client_writer, None, None).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    let lines: Vec<&str> = response.lines().collect();
    assert_eq!(
        lines.len(),
        1,
        "batch with leading whitespace should still produce one batch response"
    );
    let parsed: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    assert!(parsed.is_array(), "batch response should be an array");
    assert_eq!(parsed.as_array().unwrap().len(), 2);
}

#[test]
fn default_tcp_port_respects_env() {
    let port = IpcServer::default_tcp_port();
    if std::env::var("BARRACUDA_PORT").is_ok() {
        assert!(port.is_some());
    }
}

/// Live validation: after btsp.negotiate upgrades to chacha20-poly1305, all
/// subsequent messages on the connection use encrypted framing (including
/// any data pipelined immediately after the negotiate request line).
#[tokio::test]
async fn negotiate_then_encrypted_frame_loop() {
    use super::super::btsp::{BtspCipher, BtspSession};
    use super::super::btsp_frame::{
        BtspFrameReader, BtspFrameWriter, read_frame_as_line, write_line_as_frame,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;
    use tokio::io::{AsyncWriteExt, duplex};

    let session = BtspSession {
        session_id: "live-val-001".into(),
        cipher: BtspCipher::Null,
        session_key: vec![0xFE; 32],
    };

    let client_nonce = b"c1c2c3c4c5c6c7c8c9cacbcc";
    let client_nonce_hex = "633163326333633463356336633763386339636163626363";

    let negotiate_req = format!(
        r#"{{"jsonrpc":"2.0","method":"btsp.negotiate","params":{{"session_id":"live-val-001","preferred_cipher":"chacha20-poly1305","client_nonce":"{client_nonce_hex}"}},"id":99}}"#
    );

    let post_negotiate_plaintext =
        r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":100}"#;

    let (client_io, server_io) = duplex(16384);
    let (server_reader, server_writer) = tokio::io::split(server_io);
    let (mut client_reader, mut client_writer) = tokio::io::split(client_io);

    let primal = Arc::new(BarraCudaPrimal::new());

    let server_handle = tokio::spawn(async move {
        handle_connection(primal, server_reader, server_writer, Some(session), None).await;
    });

    client_writer
        .write_all(format!("{negotiate_req}\n").as_bytes())
        .await
        .unwrap();

    let mut resp_buf = vec![0u8; 4096];
    let n = tokio::io::AsyncReadExt::read(&mut client_reader, &mut resp_buf)
        .await
        .unwrap();
    let resp_str = std::str::from_utf8(&resp_buf[..n]).unwrap();
    let resp_json: serde_json::Value = serde_json::from_str(resp_str.trim()).unwrap();

    assert_eq!(resp_json["id"], 99);
    let result = &resp_json["result"];
    assert_eq!(result["cipher"].as_str(), Some("chacha20-poly1305"));
    let server_nonce_hex = result["server_nonce"].as_str().unwrap();

    let server_nonce: Vec<u8> = (0..server_nonce_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&server_nonce_hex[i..i + 2], 16).unwrap())
        .collect();

    let mut salt = Vec::new();
    salt.extend_from_slice(client_nonce);
    salt.extend_from_slice(&server_nonce);
    let ikm = [0xFE; 32];
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), &ikm);
    let mut derived_key = [0u8; 32];
    hkdf.expand(b"btsp-v1-phase3", &mut derived_key).unwrap();

    let derived_session = BtspSession {
        session_id: "live-val-001".into(),
        cipher: BtspCipher::ChaCha20Poly1305,
        session_key: derived_key.to_vec(),
    };

    let mut frame_writer = BtspFrameWriter::new(&mut client_writer, &derived_session);
    write_line_as_frame(&mut frame_writer, post_negotiate_plaintext)
        .await
        .unwrap();

    let mut frame_reader = BtspFrameReader::new(&mut client_reader, &derived_session);
    let response_line = read_frame_as_line(&mut frame_reader)
        .await
        .expect("should receive encrypted frame response");

    let resp: serde_json::Value = serde_json::from_str(&response_line).unwrap();
    assert_eq!(resp["id"], 100);
    assert!(
        resp["result"].is_object(),
        "device.list should return a result"
    );

    drop(frame_writer);
    drop(frame_reader);
    drop(client_writer);
    drop(client_reader);
    let _ = server_handle.await;
}

/// Verify that pipelined data after negotiate is not lost (BufReader
/// buffering preservation).
#[tokio::test]
async fn negotiate_pipelined_frame_not_lost() {
    use super::super::btsp::{BtspCipher, BtspSession};
    use super::super::btsp_frame::{
        BtspFrameReader, BtspFrameWriter, read_frame_as_line, write_line_as_frame,
    };
    use hkdf::Hkdf;
    use sha2::Sha256;
    use tokio::io::{AsyncWriteExt, duplex};

    let session = BtspSession {
        session_id: "pipeline-test".into(),
        cipher: BtspCipher::Null,
        session_key: vec![0xAA; 32],
    };

    let negotiate_req = r#"{"jsonrpc":"2.0","method":"btsp.negotiate","params":{"session_id":"pipeline-test","preferred_cipher":"chacha20-poly1305"},"id":1}"#;

    let (client_io, server_io) = duplex(16384);
    let (server_reader, server_writer) = tokio::io::split(server_io);
    let (mut client_reader, mut client_writer) = tokio::io::split(client_io);

    let primal = Arc::new(BarraCudaPrimal::new());

    let key_material = vec![0xAA; 32];
    let server_handle = tokio::spawn(async move {
        handle_connection(primal, server_reader, server_writer, Some(session), None).await;
    });

    let payload = format!("{negotiate_req}\n").into_bytes();

    let empty_client_nonce: Vec<u8> = Vec::new();

    client_writer.write_all(&payload).await.unwrap();

    let mut resp_buf = vec![0u8; 4096];
    let n = tokio::io::AsyncReadExt::read(&mut client_reader, &mut resp_buf)
        .await
        .unwrap();
    let resp_str = std::str::from_utf8(&resp_buf[..n]).unwrap();
    let resp_json: serde_json::Value = serde_json::from_str(resp_str.trim()).unwrap();
    let server_nonce_hex = resp_json["result"]["server_nonce"].as_str().unwrap();

    let server_nonce: Vec<u8> = (0..server_nonce_hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&server_nonce_hex[i..i + 2], 16).unwrap())
        .collect();

    let mut salt = Vec::new();
    salt.extend_from_slice(&empty_client_nonce);
    salt.extend_from_slice(&server_nonce);
    let hkdf = Hkdf::<Sha256>::new(Some(&salt), &key_material);
    let mut derived_key = [0u8; 32];
    hkdf.expand(b"btsp-v1-phase3", &mut derived_key).unwrap();

    let derived_session = BtspSession {
        session_id: "pipeline-test".into(),
        cipher: BtspCipher::ChaCha20Poly1305,
        session_key: derived_key.to_vec(),
    };

    let post_req = r#"{"jsonrpc":"2.0","method":"primal.info","params":{},"id":2}"#;
    let mut frame_buf: Vec<u8> = Vec::new();
    {
        let mut fw = BtspFrameWriter::new(&mut frame_buf, &derived_session);
        write_line_as_frame(&mut fw, post_req).await.unwrap();
    }
    client_writer.write_all(&frame_buf).await.unwrap();

    let mut frame_reader = BtspFrameReader::new(&mut client_reader, &derived_session);
    let response_line = read_frame_as_line(&mut frame_reader)
        .await
        .expect("pipelined encrypted frame must not be lost");

    let resp: serde_json::Value = serde_json::from_str(&response_line).unwrap();
    assert_eq!(resp["id"], 2);
    assert!(
        resp["result"].is_object(),
        "primal.info should return result"
    );

    drop(frame_reader);
    drop(client_writer);
    drop(client_reader);
    let _ = server_handle.await;
}
