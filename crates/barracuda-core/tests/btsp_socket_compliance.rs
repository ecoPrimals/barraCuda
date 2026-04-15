// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "test code uses unwrap for readability")]
//! BTSP Protocol Standard §Socket Naming compliance tests.
//!
//! Each test runs in its own process via `cargo nextest`, so `set_var` /
//! `remove_var` is safe despite the Rust 2024 `unsafe` requirement.

use barracuda_core::ipc::btsp::{BtspOutcome, guard_connection};
use barracuda_core::ipc::transport::{
    IpcServer, resolve_family_id, resolve_socket_dir, validate_insecure_guard,
};

/// Helper: clear all family-related env vars to establish a known baseline.
///
/// # Safety
///
/// Safe when each test runs in its own process (nextest default).
unsafe fn clear_family_env() {
    unsafe {
        std::env::remove_var("BARRACUDA_FAMILY_ID");
        std::env::remove_var("FAMILY_ID");
        std::env::remove_var("BIOMEOS_FAMILY_ID");
        std::env::remove_var("BIOMEOS_INSECURE");
        std::env::remove_var("BIOMEOS_SOCKET_DIR");
    }
}

// ── resolve_family_id ────────────────────────────────────────────────

#[test]
fn family_id_none_when_unset() {
    unsafe { clear_family_env() };
    assert!(resolve_family_id().is_none());
}

#[test]
fn family_id_none_when_default() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "default") };
    assert!(resolve_family_id().is_none());
}

#[test]
fn family_id_reads_barracuda_family_id_first() {
    unsafe { clear_family_env() };
    unsafe {
        std::env::set_var("BARRACUDA_FAMILY_ID", "primal-specific");
        std::env::set_var("FAMILY_ID", "composition-wide");
    }
    assert_eq!(resolve_family_id().unwrap(), "primal-specific");
}

#[test]
fn family_id_reads_family_id_when_primal_unset() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "cluster-7") };
    assert_eq!(resolve_family_id().unwrap(), "cluster-7");
}

#[test]
fn family_id_reads_legacy_biomeos_family_id() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("BIOMEOS_FAMILY_ID", "legacy-fam") };
    assert_eq!(resolve_family_id().unwrap(), "legacy-fam");
}

#[test]
fn family_id_ignores_empty_string() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "") };
    assert!(resolve_family_id().is_none());
}

// ── resolve_socket_dir ───────────────────────────────────────────────

#[test]
fn socket_dir_uses_biomeos_socket_dir_env() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("BIOMEOS_SOCKET_DIR", "/custom/sockets") };
    let dir = resolve_socket_dir();
    assert_eq!(dir, std::path::PathBuf::from("/custom/sockets"));
    unsafe { std::env::remove_var("BIOMEOS_SOCKET_DIR") };
}

#[test]
fn socket_dir_falls_back_to_xdg_biomeos() {
    unsafe { clear_family_env() };
    unsafe { std::env::remove_var("BIOMEOS_SOCKET_DIR") };
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
    unsafe { clear_family_env() };
    assert!(validate_insecure_guard().is_ok());
}

#[test]
fn insecure_guard_ok_when_insecure_only() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("BIOMEOS_INSECURE", "1") };
    assert!(validate_insecure_guard().is_ok());
}

#[test]
fn insecure_guard_ok_when_family_only() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "cluster-7") };
    assert!(validate_insecure_guard().is_ok());
}

#[test]
fn insecure_guard_rejects_family_plus_insecure() {
    unsafe { clear_family_env() };
    unsafe {
        std::env::set_var("FAMILY_ID", "cluster-7");
        std::env::set_var("BIOMEOS_INSECURE", "1");
    }
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
    unsafe { clear_family_env() };
    unsafe {
        std::env::set_var("BARRACUDA_FAMILY_ID", "prod-1");
        std::env::set_var("BIOMEOS_INSECURE", "true");
    }
    assert!(validate_insecure_guard().is_err());
}

// ── default_socket_path (family scoping) ─────────────────────────────

#[cfg(unix)]
#[test]
fn socket_path_unscoped_when_no_family() {
    unsafe { clear_family_env() };
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
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "cluster-7") };
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
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("BARRACUDA_FAMILY_ID", "override-fam") };
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
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("BIOMEOS_SOCKET_DIR", "/run/custom") };
    let path = IpcServer::default_socket_path();
    assert!(
        path.starts_with("/run/custom"),
        "expected path under /run/custom, got {}",
        path.display()
    );
    unsafe { std::env::remove_var("BIOMEOS_SOCKET_DIR") };
}

// ── BTSP Phase 2: guard_connection ──────────────────────────────────

/// Create a duplex stream pair for testing the BTSP guard without a real
/// network connection. Returns (client_half, server_half).
fn mock_stream() -> (tokio::io::DuplexStream, tokio::io::DuplexStream) {
    tokio::io::duplex(4096)
}

#[tokio::test]
async fn btsp_guard_dev_mode_when_no_family() {
    unsafe { clear_family_env() };
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
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "test-family-42") };
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
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("BARRACUDA_FAMILY_ID", "primal-override") };
    let (_client, mut server) = mock_stream();
    let outcome = guard_connection(&mut server).await;
    assert!(
        matches!(outcome, BtspOutcome::Degraded { .. }),
        "expected Degraded with primal-specific FAMILY_ID, got {outcome:?}"
    );
}

#[tokio::test]
async fn btsp_guard_degrades_on_non_hello_first_message() {
    unsafe { clear_family_env() };
    unsafe { std::env::set_var("FAMILY_ID", "test-family") };
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
