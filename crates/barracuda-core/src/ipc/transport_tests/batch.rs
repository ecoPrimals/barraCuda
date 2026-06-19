// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC batch dispatch: array semantics, empty/invalid/notification-only batches,
//! mixed request+notification, batch-via-connection-handler round-trip.

use super::*;

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
        crate::ipc::jsonrpc::INVALID_REQUEST
    );
}

#[tokio::test]
async fn test_batch_invalid_json() {
    let primal = BarraCudaPrimal::new();
    let resp = handle_batch(&primal, "[invalid")
        .await
        .expect("parse error");
    let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
    assert_eq!(parsed["error"]["code"], crate::ipc::jsonrpc::PARSE_ERROR);
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
