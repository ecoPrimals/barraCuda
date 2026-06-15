// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `ipc::transport_config` environment resolution.
//!
//! These tests manipulate process environment variables (which is `unsafe` in
//! Rust 1.86+) so they live in an integration test binary outside the
//! `#![forbid(unsafe_code)]` lib crate.
//!
//! IMPORTANT: Tests that mutate env vars must run sequentially (`--test-threads=1`)
//! or use unique keys to avoid cross-test interference. We use `serial_test`-style
//! isolation by only manipulating keys with unique prefixes per test.

use barracuda_core::ipc::transport_config;

fn with_env<F, R>(key: &str, val: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    unsafe { std::env::set_var(key, val) };
    let result = f();
    unsafe { std::env::remove_var(key) };
    result
}

fn without_env<F, R>(key: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    let prev = std::env::var(key).ok();
    unsafe { std::env::remove_var(key) };
    let result = f();
    if let Some(v) = prev {
        unsafe { std::env::set_var(key, &v) };
    }
    result
}

#[test]
fn resolve_bind_address_explicit_wins() {
    assert_eq!(
        transport_config::resolve_bind_address(Some("10.0.0.1:9999")),
        "10.0.0.1:9999"
    );
}

#[test]
fn resolve_bind_address_env_fallback() {
    with_env("BARRACUDA_IPC_BIND", "0.0.0.0:5050", || {
        assert_eq!(transport_config::resolve_bind_address(None), "0.0.0.0:5050");
    });
}

#[test]
fn resolve_bind_address_host_port_composition() {
    without_env("BARRACUDA_IPC_BIND", || {
        with_env("BARRACUDA_IPC_HOST", "192.168.1.1", || {
            with_env("BARRACUDA_IPC_PORT", "8080", || {
                assert_eq!(
                    transport_config::resolve_bind_address(None),
                    "192.168.1.1:8080"
                );
            });
        });
    });
}

#[test]
fn resolve_socket_dir_uses_env() {
    with_env("BIOMEOS_SOCKET_DIR", "/custom/sockets", || {
        assert_eq!(
            transport_config::resolve_socket_dir(),
            std::path::PathBuf::from("/custom/sockets")
        );
    });
}

#[test]
fn resolve_gate_name_defaults_unknown() {
    without_env("GATE_NAME", || {
        assert_eq!(transport_config::resolve_gate_name(), "unknown");
    });
}

#[test]
fn resolve_gate_name_uses_env() {
    with_env("GATE_NAME", "strandGate", || {
        assert_eq!(transport_config::resolve_gate_name(), "strandGate");
    });
}

#[test]
fn resolve_federation_port_default() {
    without_env("FEDERATION_PORT", || {
        assert_eq!(
            transport_config::resolve_federation_port(),
            transport_config::DEFAULT_FEDERATION_PORT
        );
    });
}

#[test]
fn resolve_federation_port_from_env() {
    with_env("FEDERATION_PORT", "9000", || {
        assert_eq!(transport_config::resolve_federation_port(), 9000);
    });
}

#[test]
fn validate_insecure_guard_rejects_family_plus_insecure() {
    with_env("BARRACUDA_FAMILY_ID", "prod", || {
        with_env("BIOMEOS_INSECURE", "1", || {
            assert!(transport_config::validate_insecure_guard().is_err());
        });
    });
}

#[test]
fn validate_insecure_guard_ok_without_family() {
    without_env("BARRACUDA_FAMILY_ID", || {
        without_env("FAMILY_ID", || {
            without_env("BIOMEOS_FAMILY_ID", || {
                with_env("BIOMEOS_INSECURE", "1", || {
                    assert!(transport_config::validate_insecure_guard().is_ok());
                });
            });
        });
    });
}
