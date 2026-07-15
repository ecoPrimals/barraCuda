// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "test code uses unwrap for readability")]
//! BTSP Protocol Standard §Socket Naming compliance tests.
//!
//! Each test runs in its own process via `cargo nextest`, so `set_var` /
//! `remove_var` is safe despite the Rust 2024 `unsafe` requirement.
//! The `env_set` / `env_remove` helpers encapsulate the safety invariant.

use barracuda_core::ipc::btsp::{BtspOutcome, guard_connection};
use barracuda_core::ipc::transport::{
    IpcServer, resolve_family_id, resolve_socket_dir, validate_insecure_guard,
};

/// Set an environment variable. Safe under nextest (one test per process).
fn env_set(key: &str, val: &str) {
    // SAFETY: nextest runs each test in its own process, so no concurrent
    // threads are reading environment variables.
    unsafe { std::env::set_var(key, val) };
}

/// Remove an environment variable. Safe under nextest (one test per process).
fn env_remove(key: &str) {
    // SAFETY: nextest runs each test in its own process.
    unsafe { std::env::remove_var(key) };
}

/// Clear all family-related env vars to establish a known baseline.
fn clear_family_env() {
    env_remove("BARRACUDA_FAMILY_ID");
    env_remove("FAMILY_ID");
    env_remove("BIOMEOS_FAMILY_ID");
    env_remove("BIOMEOS_INSECURE");
    env_remove("BIOMEOS_SOCKET_DIR");
}

// ── resolve_family_id ────────────────────────────────────────────────

#[test]
fn family_id_none_when_unset() {
    clear_family_env();
    assert!(resolve_family_id().is_none());
}

#[test]
fn family_id_none_when_default() {
    clear_family_env();
    env_set("FAMILY_ID", "default");
    assert!(resolve_family_id().is_none());
}

#[test]
fn family_id_reads_barracuda_family_id_first() {
    clear_family_env();
    env_set("BARRACUDA_FAMILY_ID", "primal-specific");
    env_set("FAMILY_ID", "composition-wide");
    assert_eq!(resolve_family_id().unwrap(), "primal-specific");
}

#[test]
fn family_id_reads_family_id_when_primal_unset() {
    clear_family_env();
    env_set("FAMILY_ID", "cluster-7");
    assert_eq!(resolve_family_id().unwrap(), "cluster-7");
}

#[test]
fn family_id_reads_legacy_biomeos_family_id() {
    clear_family_env();
    env_set("BIOMEOS_FAMILY_ID", "legacy-fam");
    assert_eq!(resolve_family_id().unwrap(), "legacy-fam");
}

#[test]
fn family_id_ignores_empty_string() {
    clear_family_env();
    env_set("FAMILY_ID", "");
    assert!(resolve_family_id().is_none());
}

// ── resolve_socket_dir ───────────────────────────────────────────────

#[test]
fn socket_dir_uses_biomeos_socket_dir_env() {
    clear_family_env();
    env_set("BIOMEOS_SOCKET_DIR", "/custom/sockets");
    let dir = resolve_socket_dir();
    assert_eq!(dir, std::path::PathBuf::from("/custom/sockets"));
    env_remove("BIOMEOS_SOCKET_DIR");
}

#[test]
fn socket_dir_falls_back_to_xdg_biomeos() {
    clear_family_env();
    env_remove("BIOMEOS_SOCKET_DIR");
    let dir = resolve_socket_dir();
    let s = dir.to_string_lossy();
    assert!(
        s.contains("biomeos"),
        "fallback should contain 'biomeos', got {s}"
    );
}

// ── validate_insecure_guard ──────────────────────────────────────────

#[test]
fn insecure_guard_ok_when_both_unset() {
    clear_family_env();
    assert!(validate_insecure_guard().is_ok());
}

#[test]
fn insecure_guard_ok_when_insecure_only() {
    clear_family_env();
    env_set("BIOMEOS_INSECURE", "1");
    assert!(validate_insecure_guard().is_ok());
}

#[test]
fn insecure_guard_ok_when_family_only() {
    clear_family_env();
    env_set("FAMILY_ID", "cluster-7");
    assert!(validate_insecure_guard().is_ok());
}

#[test]
fn insecure_guard_rejects_family_plus_insecure() {
    clear_family_env();
    env_set("FAMILY_ID", "cluster-7");
    env_set("BIOMEOS_INSECURE", "1");
    let err = validate_insecure_guard().unwrap_err().to_string();
    assert!(
        err.contains("cluster-7"),
        "error should mention the family ID: {err}"
    );
    assert!(
        err.contains("BTSP"),
        "error should reference BTSP standard: {err}"
    );
}

#[test]
fn insecure_guard_rejects_with_true_string() {
    clear_family_env();
    env_set("BARRACUDA_FAMILY_ID", "prod-1");
    env_set("BIOMEOS_INSECURE", "true");
    assert!(validate_insecure_guard().is_err());
}

// ── default_socket_path (family scoping) ─────────────────────────────

#[cfg(unix)]
#[test]
fn socket_path_unscoped_when_no_family() {
    clear_family_env();
    let path = IpcServer::default_socket_path();
    assert!(
        path.to_string_lossy().ends_with("math.sock"),
        "expected math.sock, got {}",
        path.display()
    );
}

#[cfg(unix)]
#[test]
fn socket_path_scoped_with_family_id() {
    clear_family_env();
    env_set("FAMILY_ID", "cluster-7");
    let path = IpcServer::default_socket_path();
    assert!(
        path.to_string_lossy().ends_with("math-cluster-7.sock"),
        "expected math-cluster-7.sock, got {}",
        path.display()
    );
}

#[cfg(unix)]
#[test]
fn socket_path_scoped_with_primal_specific_family() {
    clear_family_env();
    env_set("BARRACUDA_FAMILY_ID", "override-fam");
    let path = IpcServer::default_socket_path();
    assert!(
        path.to_string_lossy().ends_with("math-override-fam.sock"),
        "expected math-override-fam.sock, got {}",
        path.display()
    );
}

#[cfg(unix)]
#[test]
fn socket_path_respects_biomeos_socket_dir() {
    clear_family_env();
    env_set("BIOMEOS_SOCKET_DIR", "/run/custom");
    let path = IpcServer::default_socket_path();
    assert!(
        path.starts_with("/run/custom"),
        "expected path under /run/custom, got {}",
        path.display()
    );
    env_remove("BIOMEOS_SOCKET_DIR");
}

// ── BTSP Phase 2: guard_connection ──────────────────────────────────

/// Create a duplex stream pair for testing the BTSP guard without a real
/// network connection. Returns (client_half, server_half).
fn mock_stream() -> (tokio::io::DuplexStream, tokio::io::DuplexStream) {
    tokio::io::duplex(4096)
}

#[tokio::test]
async fn btsp_guard_dev_mode_when_no_family() {
    clear_family_env();
    let (_client, mut server) = mock_stream();
    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(outcome, BtspOutcome::DevMode),
        "expected DevMode, got {outcome:?}"
    );
    assert!(outcome.should_accept());
}

#[tokio::test]
async fn btsp_guard_degrades_when_provider_absent() {
    clear_family_env();
    env_set("FAMILY_ID", "test-family-42");
    let (_client, mut server) = mock_stream();
    // Client side doesn't send anything → timeout → legacy fallback → Degraded
    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(outcome, BtspOutcome::Degraded { .. }),
        "expected Degraded (security provider absent / no ClientHello), got {outcome:?}"
    );
    assert!(
        outcome.should_accept(),
        "degraded mode should still accept connections"
    );
}

#[tokio::test]
async fn btsp_guard_degrades_with_primal_specific_family() {
    clear_family_env();
    env_set("BARRACUDA_FAMILY_ID", "primal-override");
    let (_client, mut server) = mock_stream();
    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(outcome, BtspOutcome::Degraded { .. }),
        "expected Degraded with primal-specific FAMILY_ID, got {outcome:?}"
    );
}

#[tokio::test]
async fn btsp_guard_degrades_on_non_hello_first_message() {
    clear_family_env();
    env_set("FAMILY_ID", "test-family");
    let (mut client, mut server) = mock_stream();
    // Client sends a JSON-RPC request instead of ClientHello → legacy fallback
    tokio::spawn(async move {
        use tokio::io::AsyncWriteExt;
        let msg = r#"{"jsonrpc":"2.0","method":"health.check","id":1}"#;
        client.write_all(msg.as_bytes()).await.unwrap();
        client.write_all(b"\n").await.unwrap();
    });
    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(outcome, BtspOutcome::Degraded { .. }),
        "expected Degraded for non-BTSP client, got {outcome:?}"
    );
}

#[test]
fn btsp_outcome_rejected_refuses_connection() {
    let outcome = BtspOutcome::Rejected {
        reason: "challenge failed".to_string(),
    };
    assert!(
        !outcome.should_accept(),
        "rejected outcome should refuse connection"
    );
}

// ── Full relay integration (mock security provider, Unix-only) ───────

#[cfg(unix)]
/// Spawn a mock security-domain provider that accepts one connection and
/// handles `btsp.session.create` + `btsp.session.verify` sequentially
/// on the same stream (per SOURDOUGH_BTSP_RELAY_PATTERN).
async fn spawn_mock_provider(sock_path: &std::path::Path) -> tokio::task::JoinHandle<()> {
    let listener = tokio::net::UnixListener::bind(sock_path).unwrap();
    tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let (stream, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(stream);

        for id in [1u64, 2] {
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let req: serde_json::Value = serde_json::from_str(&line).unwrap();

            let method = req["method"].as_str().unwrap();
            let response = match method {
                "btsp.session.create" => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "session_token": "tok-abc123",
                        "server_ephemeral_pub": "mock_server_pub_b64==",
                        "challenge": "mock_challenge_b64=="
                    }
                }),
                "btsp.session.verify" => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "verified": true,
                        "cipher": "none",
                        "session_key": ""
                    }
                }),
                other => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "error": { "code": -32601, "message": format!("unknown method: {other}") }
                }),
            };
            let mut resp_line = serde_json::to_string(&response).unwrap();
            resp_line.push('\n');
            reader
                .get_mut()
                .write_all(resp_line.as_bytes())
                .await
                .unwrap();
            reader.get_mut().flush().await.unwrap();
        }
    })
}

#[cfg(unix)]
#[tokio::test]
async fn btsp_full_relay_authenticated_null_cipher() {
    let tmp = tempfile::tempdir().unwrap();
    let provider_sock = tmp.path().join("provider.sock");

    clear_family_env();
    env_set("FAMILY_ID", "relay-test-family");
    env_set("FAMILY_SEED", "test-seed-value");
    env_set("BTSP_PROVIDER_SOCKET", provider_sock.to_str().unwrap());

    let provider_handle = spawn_mock_provider(&provider_sock).await;
    // Small yield to let the listener bind
    tokio::task::yield_now().await;

    let (mut client, mut server) = tokio::io::duplex(8192);

    let client_task = tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let hello = serde_json::json!({
            "type": "ClientHello",
            "version": 1,
            "client_ephemeral_pub": "mock_client_pub_b64=="
        });
        let mut line = serde_json::to_string(&hello).unwrap();
        line.push('\n');
        client.write_all(line.as_bytes()).await.unwrap();
        client.flush().await.unwrap();

        let mut reader = BufReader::new(client);
        let mut server_hello_line = String::new();
        reader.read_line(&mut server_hello_line).await.unwrap();
        let server_hello: serde_json::Value = serde_json::from_str(&server_hello_line).unwrap();
        assert_eq!(server_hello["type"], "ServerHello");
        assert_eq!(
            server_hello["server_ephemeral_pub"],
            "mock_server_pub_b64=="
        );
        assert_eq!(server_hello["challenge"], "mock_challenge_b64==");

        let challenge_resp = serde_json::json!({
            "type": "ChallengeResponse",
            "response": "mock_hmac_b64==",
            "preferred_cipher": "none"
        });
        let mut resp_line = serde_json::to_string(&challenge_resp).unwrap();
        resp_line.push('\n');
        reader
            .get_mut()
            .write_all(resp_line.as_bytes())
            .await
            .unwrap();
        reader.get_mut().flush().await.unwrap();

        let mut complete_line = String::new();
        reader.read_line(&mut complete_line).await.unwrap();
        let complete: serde_json::Value = serde_json::from_str(&complete_line).unwrap();
        assert_eq!(complete["type"], "HandshakeComplete");
        assert_eq!(complete["session_id"], "tok-abc123");
    });

    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(&outcome, BtspOutcome::Authenticated(session) if session.cipher == barracuda_core::ipc::btsp::BtspCipher::Null),
        "expected Authenticated(Null cipher), got {outcome:?}"
    );
    let session = outcome.session().unwrap();
    assert_eq!(session.session_id, "tok-abc123");
    assert!(session.session_key.is_empty());

    client_task.await.unwrap();
    provider_handle.await.unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn btsp_full_relay_rejected_by_provider() {
    let tmp = tempfile::tempdir().unwrap();
    let provider_sock = tmp.path().join("provider-reject.sock");

    clear_family_env();
    env_set("FAMILY_ID", "reject-test-family");
    env_set("FAMILY_SEED", "test-seed");
    env_set("BTSP_PROVIDER_SOCKET", provider_sock.to_str().unwrap());

    let listener = tokio::net::UnixListener::bind(&provider_sock).unwrap();
    let provider_handle = tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let (stream, _) = listener.accept().await.unwrap();
        let mut reader = BufReader::new(stream);

        // session.create succeeds
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        let resp = serde_json::json!({
            "jsonrpc": "2.0", "id": 1,
            "result": {
                "session_token": "tok-rej",
                "server_ephemeral_pub": "srv_pub==",
                "challenge": "chal=="
            }
        });
        let mut resp_line = serde_json::to_string(&resp).unwrap();
        resp_line.push('\n');
        reader
            .get_mut()
            .write_all(resp_line.as_bytes())
            .await
            .unwrap();
        reader.get_mut().flush().await.unwrap();

        // session.verify rejects
        let mut line2 = String::new();
        reader.read_line(&mut line2).await.unwrap();
        let reject = serde_json::json!({
            "jsonrpc": "2.0", "id": 2,
            "result": {
                "verified": false,
                "reason": "HMAC mismatch"
            }
        });
        let mut rej_line = serde_json::to_string(&reject).unwrap();
        rej_line.push('\n');
        reader
            .get_mut()
            .write_all(rej_line.as_bytes())
            .await
            .unwrap();
        reader.get_mut().flush().await.unwrap();
    });

    tokio::task::yield_now().await;

    let (mut client, mut server) = tokio::io::duplex(8192);
    tokio::spawn(async move {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        let hello = serde_json::json!({
            "type": "ClientHello",
            "client_ephemeral_pub": "client_pub=="
        });
        let mut line = serde_json::to_string(&hello).unwrap();
        line.push('\n');
        client.write_all(line.as_bytes()).await.unwrap();
        client.flush().await.unwrap();

        let mut reader = BufReader::new(client);
        let mut sh = String::new();
        reader.read_line(&mut sh).await.unwrap();

        let resp = serde_json::json!({
            "type": "ChallengeResponse",
            "response": "bad_hmac==",
            "preferred_cipher": "none"
        });
        let mut line = serde_json::to_string(&resp).unwrap();
        line.push('\n');
        reader.get_mut().write_all(line.as_bytes()).await.unwrap();
        reader.get_mut().flush().await.unwrap();

        // Connection may close — that's expected
        let _ = reader.read_line(&mut String::new()).await;
    });

    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(&outcome, BtspOutcome::Rejected { reason } if reason.contains("HMAC")),
        "expected Rejected, got {outcome:?}"
    );

    provider_handle.await.unwrap();
}
