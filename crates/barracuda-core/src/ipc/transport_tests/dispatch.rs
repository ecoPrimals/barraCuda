// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC single-line dispatch: valid requests, parse errors, notifications.

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
    assert_eq!(resp.error.unwrap().code, crate::ipc::jsonrpc::PARSE_ERROR);
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
