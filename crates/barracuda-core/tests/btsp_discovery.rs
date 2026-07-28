// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `ipc::btsp_discovery` file-based resolution.
//!
//! These tests manipulate process environment variables (which is `unsafe` in
//! Rust 1.86+) so they live in an integration test binary outside the
//! `#![forbid(unsafe_code)]` lib crate.
//!
//! Unix-only: discovery relies on Unix domain sockets for provider probing.
#![cfg(unix)]
#![expect(clippy::unwrap_used, reason = "test assertions")]

use barracuda_core::env_keys;
use std::io::Write;

static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn discover_by_capability_from_discovery_file() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("crypto.sock");
    std::os::unix::net::UnixListener::bind(&sock_path).unwrap();

    let discovery_file = dir.path().join("crypto.json");
    let content = serde_json::json!({
        "name": "test-crypto-provider",
        "methods": ["btsp.session.create", "btsp.session.verify"],
        "transports": {
            "unix": format!("unix://{}", sock_path.display())
        }
    });
    let mut f = std::fs::File::create(&discovery_file).unwrap();
    write!(f, "{content}").unwrap();

    #[expect(deprecated, reason = "clearing legacy env for test isolation")]
    let legacy = env_keys::BEARDOG_SOCKET;
    // SAFETY: ENV_MUTEX serializes env mutation across tests.
    unsafe { std::env::set_var(env_keys::BIOMEOS_SOCKET_DIR, dir.path().as_os_str()) };
    unsafe { std::env::remove_var(env_keys::BTSP_PROVIDER_SOCKET) };
    unsafe { std::env::remove_var(legacy) };
    unsafe { std::env::remove_var(env_keys::BARRACUDA_FAMILY_ID) };
    unsafe { std::env::remove_var(env_keys::FAMILY_ID) };
    unsafe { std::env::remove_var(env_keys::BIOMEOS_FAMILY_ID) };

    let result = barracuda_core::ipc::btsp_discovery::discover_security_provider();
    assert_eq!(result, Some(sock_path));

    unsafe { std::env::remove_var(env_keys::BIOMEOS_SOCKET_DIR) };
}

#[test]
fn discover_security_provider_uses_env_override() {
    let _lock = ENV_MUTEX.lock().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("my-security.sock");
    std::os::unix::net::UnixListener::bind(&sock_path).unwrap();

    // SAFETY: ENV_MUTEX serializes env mutation across tests.
    unsafe { std::env::set_var(env_keys::BTSP_PROVIDER_SOCKET, sock_path.as_os_str()) };

    let result = barracuda_core::ipc::btsp_discovery::discover_security_provider();
    assert_eq!(result, Some(sock_path));

    unsafe { std::env::remove_var(env_keys::BTSP_PROVIDER_SOCKET) };
}

#[test]
fn discover_security_provider_none_when_nothing_available() {
    let _lock = ENV_MUTEX.lock().unwrap();
    #[expect(deprecated, reason = "clearing legacy env for test isolation")]
    let legacy = env_keys::BEARDOG_SOCKET;
    // SAFETY: ENV_MUTEX serializes env mutation across tests.
    unsafe { std::env::remove_var(env_keys::BTSP_PROVIDER_SOCKET) };
    unsafe { std::env::remove_var(legacy) };
    unsafe { std::env::remove_var(env_keys::BARRACUDA_FAMILY_ID) };
    unsafe { std::env::remove_var(env_keys::FAMILY_ID) };
    unsafe { std::env::remove_var(env_keys::BIOMEOS_FAMILY_ID) };
    unsafe { std::env::set_var(env_keys::BIOMEOS_SOCKET_DIR, "/nonexistent_dir_for_test") };

    let result = barracuda_core::ipc::btsp_discovery::discover_security_provider();
    assert_eq!(result, None);

    unsafe { std::env::remove_var(env_keys::BIOMEOS_SOCKET_DIR) };
}
