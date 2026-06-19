// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport configuration and environment resolution.
//!
//! Standalone utilities for resolving bind addresses, socket paths, and
//! ecosystem discovery directories from environment variables per
//! wateringHole standards (`PRIMAL_SELF_KNOWLEDGE_STANDARD.md`,
//! `BTSP_PROTOCOL_STANDARD.md`, `ECOBIN_ARCHITECTURE_STANDARD.md`).

use crate::env_keys;

/// Default TCP bind host (`127.0.0.1` = localhost-only).
pub const DEFAULT_BIND_HOST: &str = "127.0.0.1";

/// Default family ID when no `FAMILY_ID` env var is set.
const DEFAULT_FAMILY_ID: &str = "default";

/// Ecosystem socket namespace. Override via `BIOMEOS_SOCKET_DIR`.
pub const DEFAULT_ECOSYSTEM_SOCKET_DIR: &str = "biomeos";

/// Resolve TCP bind host: `BARRACUDA_IPC_HOST` → [`DEFAULT_BIND_HOST`].
pub fn resolve_bind_host() -> String {
    std::env::var(env_keys::BARRACUDA_IPC_HOST).unwrap_or_else(|_| DEFAULT_BIND_HOST.to_string())
}

/// Resolve the family ID per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §4.
///
/// Precedence: `BARRACUDA_FAMILY_ID` → `FAMILY_ID` → `BIOMEOS_FAMILY_ID` (legacy).
/// Returns `None` when unset or `"default"`.
pub fn resolve_family_id() -> Option<String> {
    const KEYS: &[&str] = &[
        env_keys::BARRACUDA_FAMILY_ID,
        env_keys::FAMILY_ID,
        env_keys::BIOMEOS_FAMILY_ID,
    ];
    for key in KEYS {
        if let Ok(val) = std::env::var(key)
            && !val.is_empty()
            && val != DEFAULT_FAMILY_ID
        {
            return Some(val);
        }
    }
    None
}

/// Resolve the socket directory per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3.
///
/// Resolution: `BIOMEOS_SOCKET_DIR` → `$XDG_RUNTIME_DIR/biomeos` → `$TMPDIR/biomeos`.
pub fn resolve_socket_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var(env_keys::BIOMEOS_SOCKET_DIR) {
        return std::path::PathBuf::from(dir);
    }
    let base = std::env::var(env_keys::XDG_RUNTIME_DIR)
        .map_or_else(|_| std::env::temp_dir(), std::path::PathBuf::from);
    base.join(DEFAULT_ECOSYSTEM_SOCKET_DIR)
}

/// Validates that `FAMILY_ID` + `BIOMEOS_INSECURE=1` are never both set.
///
/// Per `BTSP_PROTOCOL_STANDARD.md` §Compliance: you cannot claim a family
/// AND skip authentication.
pub fn validate_insecure_guard() -> crate::error::Result<()> {
    let family_id = resolve_family_id();
    let insecure = std::env::var(env_keys::BIOMEOS_INSECURE)
        .is_ok_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    if let Some(ref fid) = family_id
        && insecure
    {
        return Err(crate::error::BarracudaCoreError::lifecycle(format!(
            "FAMILY_ID={fid} but BIOMEOS_INSECURE=1 — cannot claim a family \
                 and skip authentication. Unset one or the other. \
                 See BTSP_PROTOCOL_STANDARD.md §Compliance."
        )));
    }
    Ok(())
}

/// Resolve TCP bind address: `explicit` → `BARRACUDA_IPC_BIND` →
/// `BARRACUDA_IPC_HOST:BARRACUDA_IPC_PORT` → `127.0.0.1:0`.
pub fn resolve_bind_address(explicit: Option<&str>) -> String {
    if let Some(addr) = explicit {
        return addr.to_string();
    }
    if let Ok(addr) = std::env::var(env_keys::BARRACUDA_IPC_BIND) {
        return addr;
    }
    let host = std::env::var(env_keys::BARRACUDA_IPC_HOST)
        .unwrap_or_else(|_| DEFAULT_BIND_HOST.to_string());
    std::env::var(env_keys::BARRACUDA_IPC_PORT)
        .map_or_else(|_| format!("{host}:0"), |port| format!("{host}:{port}"))
}

/// Return the canonical UDS path that biomeOS uses to reach this primal.
///
/// Used in `primal.announce` so biomeOS can route `capability.call` traffic
/// directly. Format: `$XDG_RUNTIME_DIR/biomeos/{domain}[-{family}].sock`.
#[cfg(unix)]
#[must_use]
pub fn discovery_socket_path() -> String {
    let dir = resolve_socket_dir();
    let domain = crate::PRIMAL_DOMAIN;
    let sock_name = match resolve_family_id() {
        Some(family_id) => format!("{domain}-{family_id}.sock"),
        None => format!("{domain}.sock"),
    };
    dir.join(sock_name).to_string_lossy().into_owned()
}

#[cfg(not(unix))]
#[must_use]
pub fn discovery_socket_path() -> String {
    String::from("unsupported")
}

/// Resolve the gate name this primal is deployed on.
///
/// Discovered at runtime via `GATE_NAME` env var. Returns `"unknown"` if unset.
/// Primals do not hardcode their gate identity — they discover it at runtime.
#[must_use]
pub fn resolve_gate_name() -> String {
    std::env::var(env_keys::GATE_NAME).unwrap_or_else(|_| "unknown".into())
}

/// Default federation port for Songbird mesh.
pub const DEFAULT_FEDERATION_PORT: u16 = 7700;

/// Resolve federation port from environment or default.
#[must_use]
pub fn resolve_federation_port() -> u16 {
    std::env::var(env_keys::FEDERATION_PORT)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_FEDERATION_PORT)
}

/// Socket filename prefix for the security provider (e.g. bearDog).
/// Used for health probing — we look for `{prefix}*.sock` in the socket dir.
pub const SECURITY_PROVIDER_SOCKET_PREFIX: &str = "beardog";

/// Socket filename prefix for the discovery service (e.g. Songbird).
pub const DISCOVERY_SOCKET_PREFIX: &str = "songbird";
