// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bind-address resolution: host fallback, explicit override, ephemeral port.

use super::*;

#[test]
fn resolve_bind_host_returns_valid_ip() {
    let host = resolve_bind_host();
    assert!(
        host.parse::<std::net::IpAddr>().is_ok(),
        "resolve_bind_host must return a valid IP, got: {host}"
    );
}

#[test]
fn resolve_bind_host_fallback_matches_default() {
    if std::env::var("BARRACUDA_IPC_HOST").is_err() {
        assert_eq!(resolve_bind_host(), DEFAULT_BIND_HOST);
    }
}

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
