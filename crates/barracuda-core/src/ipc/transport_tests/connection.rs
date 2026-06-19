// SPDX-License-Identifier: AGPL-3.0-or-later
//! Connection handler integration tests: request/response, multi-line, notifications,
//! TCP bind, tarpc fallback, whitespace tolerance, replayed lines.

use super::*;

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
