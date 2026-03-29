// SPDX-License-Identifier: AGPL-3.0-or-later

//! IPC tolerances — env-configurable timeouts for primal discovery and RPC.
//!
//! All values can be overridden via environment variables for deployment
//! tuning without recompilation. Absorbed from ludoSpring V34.

const DEFAULT_RPC_TIMEOUT_SECS: u64 = 5;
const DEFAULT_PROBE_TIMEOUT_MS: u64 = 500;
const DEFAULT_CONNECT_PROBE_TIMEOUT_MS: u64 = 200;

fn env_u64(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Timeout for JSON-RPC calls to peer primals (seconds).
///
/// Override: `BARRACUDA_RPC_TIMEOUT_SECS`
///
/// Default 5s accommodates cold startup of AI providers while failing
/// fast on network issues.
#[must_use]
pub fn rpc_timeout_secs() -> u64 {
    env_u64("BARRACUDA_RPC_TIMEOUT_SECS", DEFAULT_RPC_TIMEOUT_SECS)
}

/// Probe timeout for socket capability verification (milliseconds).
///
/// Override: `BARRACUDA_PROBE_TIMEOUT_MS`
///
/// Default 500ms is enough for a local Unix socket round-trip including
/// `lifecycle.status` parsing.
#[must_use]
pub fn probe_timeout_ms() -> u64 {
    env_u64("BARRACUDA_PROBE_TIMEOUT_MS", DEFAULT_PROBE_TIMEOUT_MS)
}

/// Connect-probe timeout for quick liveness checks (milliseconds).
///
/// Override: `BARRACUDA_CONNECT_PROBE_TIMEOUT_MS`
///
/// Default 200ms is generous for loopback connections.
#[must_use]
pub fn connect_probe_timeout_ms() -> u64 {
    env_u64(
        "BARRACUDA_CONNECT_PROBE_TIMEOUT_MS",
        DEFAULT_CONNECT_PROBE_TIMEOUT_MS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_reasonable() {
        assert_eq!(rpc_timeout_secs(), 5);
        assert_eq!(probe_timeout_ms(), 500);
        assert_eq!(connect_probe_timeout_ms(), 200);
    }

    #[test]
    fn env_override_works() {
        assert_eq!(env_u64("NONEXISTENT_VAR_12345", 42), 42);
    }
}
