// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration tests for `ipc::transport_config` environment resolution.
//!
//! These tests manipulate process environment variables (which is `unsafe` in
//! Rust 1.86+) so they live in an integration test binary outside the
//! `#![forbid(unsafe_code)]` lib crate.
//!
//! All env-mutating tests hold `ENV_MUTEX` to prevent races under `cargo test`.
#![expect(clippy::unwrap_used, reason = "test assertions")]

use barracuda_core::ipc::transport_config;

static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_env<F, R>(key: &str, val: &str, f: F) -> R
where
    F: FnOnce() -> R,
{
    // SAFETY: caller holds ENV_MUTEX, so no concurrent env mutation.
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
    // SAFETY: caller holds ENV_MUTEX, so no concurrent env mutation.
    unsafe { std::env::remove_var(key) };
    let result = f();
    if let Some(v) = prev {
        unsafe { std::env::set_var(key, &v) };
    }
    result
}

#[test]
fn resolve_bind_address_explicit_wins() {
    let _lock = ENV_MUTEX.lock().unwrap();
    assert_eq!(
        transport_config::resolve_bind_address(Some("10.0.0.1:9999")),
        "10.0.0.1:9999"
    );
}

#[test]
fn resolve_bind_address_env_fallback() {
    let _lock = ENV_MUTEX.lock().unwrap();
    with_env("BARRACUDA_IPC_BIND", "0.0.0.0:5050", || {
        assert_eq!(transport_config::resolve_bind_address(None), "0.0.0.0:5050");
    });
}

#[test]
fn resolve_bind_address_host_port_composition() {
    let _lock = ENV_MUTEX.lock().unwrap();
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
    let _lock = ENV_MUTEX.lock().unwrap();
    with_env("BIOMEOS_SOCKET_DIR", "/custom/sockets", || {
        assert_eq!(
            transport_config::resolve_socket_dir(),
            std::path::PathBuf::from("/custom/sockets")
        );
    });
}

#[test]
fn resolve_gate_name_defaults_unknown() {
    let _lock = ENV_MUTEX.lock().unwrap();
    without_env("GATE_NAME", || {
        assert_eq!(transport_config::resolve_gate_name(), "unknown");
    });
}

#[test]
fn resolve_gate_name_uses_env() {
    let _lock = ENV_MUTEX.lock().unwrap();
    with_env("GATE_NAME", "strandGate", || {
        assert_eq!(transport_config::resolve_gate_name(), "strandGate");
    });
}

#[test]
fn resolve_federation_port_default() {
    let _lock = ENV_MUTEX.lock().unwrap();
    without_env("FEDERATION_PORT", || {
        assert_eq!(
            transport_config::resolve_federation_port(),
            transport_config::DEFAULT_FEDERATION_PORT
        );
    });
}

#[test]
fn resolve_federation_port_from_env() {
    let _lock = ENV_MUTEX.lock().unwrap();
    with_env("FEDERATION_PORT", "9000", || {
        assert_eq!(transport_config::resolve_federation_port(), 9000);
    });
}

#[test]
fn validate_insecure_guard_rejects_family_plus_insecure() {
    let _lock = ENV_MUTEX.lock().unwrap();
    with_env("BARRACUDA_FAMILY_ID", "prod", || {
        with_env("BIOMEOS_INSECURE", "1", || {
            assert!(transport_config::validate_insecure_guard().is_err());
        });
    });
}

#[test]
fn validate_insecure_guard_ok_without_family() {
    let _lock = ENV_MUTEX.lock().unwrap();
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
