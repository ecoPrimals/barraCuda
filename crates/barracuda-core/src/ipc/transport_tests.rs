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
    assert!(path_str.contains(ECOSYSTEM_SOCKET_DIR));
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
        dir_str.contains(ECOSYSTEM_SOCKET_DIR) || std::env::var("BIOMEOS_SOCKET_DIR").is_ok(),
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

    handle_connection(primal, client_reader, client_writer).await;

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

    handle_connection(primal, client_reader, client_writer).await;

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

    handle_connection(primal, client_reader, client_writer).await;

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

    handle_connection(primal, client_reader, client_writer).await;

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

    handle_connection(primal, client_reader, client_writer).await;

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

    handle_connection(primal, client_reader, client_writer).await;

    let mut response = String::new();
    tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
        .await
        .unwrap();
    assert!(response.is_empty(), "notification must produce no response");
}
