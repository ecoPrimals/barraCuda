// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `ipc::btsp_discovery` file-based resolution.
//!
//! These tests manipulate process environment variables (which is `unsafe` in
//! Rust 1.86+) so they live in an integration test binary outside the
//! `#![forbid(unsafe_code)]` lib crate.
#![expect(clippy::unwrap_used, reason = "test assertions")]

use std::io::Write;

#[test]
fn discover_by_capability_from_discovery_file() {
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

    unsafe { std::env::set_var("BIOMEOS_SOCKET_DIR", dir.path().as_os_str()) };
    unsafe { std::env::remove_var("BTSP_PROVIDER_SOCKET") };
    unsafe { std::env::remove_var("BEARDOG_SOCKET") };
    unsafe { std::env::remove_var("BARRACUDA_FAMILY_ID") };
    unsafe { std::env::remove_var("FAMILY_ID") };
    unsafe { std::env::remove_var("BIOMEOS_FAMILY_ID") };

    let result = barracuda_core::ipc::btsp_discovery::discover_security_provider();
    assert_eq!(result, Some(sock_path));

    unsafe { std::env::remove_var("BIOMEOS_SOCKET_DIR") };
}

#[test]
fn discover_security_provider_uses_env_override() {
    let dir = tempfile::tempdir().unwrap();
    let sock_path = dir.path().join("my-security.sock");
    std::os::unix::net::UnixListener::bind(&sock_path).unwrap();

    unsafe { std::env::set_var("BTSP_PROVIDER_SOCKET", sock_path.as_os_str()) };

    let result = barracuda_core::ipc::btsp_discovery::discover_security_provider();
    assert_eq!(result, Some(sock_path));

    unsafe { std::env::remove_var("BTSP_PROVIDER_SOCKET") };
}

#[test]
fn discover_security_provider_none_when_nothing_available() {
    unsafe { std::env::remove_var("BTSP_PROVIDER_SOCKET") };
    unsafe { std::env::remove_var("BEARDOG_SOCKET") };
    unsafe { std::env::remove_var("BARRACUDA_FAMILY_ID") };
    unsafe { std::env::remove_var("FAMILY_ID") };
    unsafe { std::env::remove_var("BIOMEOS_FAMILY_ID") };
    unsafe { std::env::set_var("BIOMEOS_SOCKET_DIR", "/nonexistent_dir_for_test") };

    let result = barracuda_core::ipc::btsp_discovery::discover_security_provider();
    assert_eq!(result, None);

    unsafe { std::env::remove_var("BIOMEOS_SOCKET_DIR") };
}
