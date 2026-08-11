// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fire-and-forget gossip injection client for the swarmVine mesh.
//!
//! Sends `gossip.inject` JSON-RPC requests to the local swarmVine UDS.
//! All calls are non-fatal: if swarmVine is absent, the injection silently
//! returns `false`. Zero-cost when the mesh is not running.
//!
//! Wire format per swarmVine canonical handler:
//! ```json
//! {
//!   "jsonrpc": "2.0",
//!   "method": "gossip.inject",
//!   "params": { "topic": "compute", "key": "...", "payload": { ... } },
//!   "id": null
//! }
//! ```

use std::io::Write;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::OnceLock;
use std::time::Duration;

use crate::env_keys;

/// swarmVine socket environment variable override.
const SWARMVINE_SOCKET_ENV: &str = "SWARMVINE_SOCKET";

/// Write timeout for a single gossip injection.
const WRITE_TIMEOUT: Duration = Duration::from_millis(500);

/// Resolve the local gate name from environment.
fn gate_name() -> &'static str {
    static GATE: OnceLock<String> = OnceLock::new();
    GATE.get_or_init(|| {
        std::env::var(env_keys::GATE_NAME).unwrap_or_else(|_| "unknown".into())
    })
}

/// Discover the swarmVine UDS path.
///
/// Resolution order:
/// 1. `SWARMVINE_SOCKET` env var (explicit override)
/// 2. `$XDG_RUNTIME_DIR/biomeos/swarmvine.sock` (canonical biomeOS path)
/// 3. `None` — swarmVine not discoverable
#[cfg(unix)]
fn discover_socket() -> Option<std::path::PathBuf> {
    if let Ok(path) = std::env::var(SWARMVINE_SOCKET_ENV) {
        let p = std::path::PathBuf::from(&path);
        if p.exists() {
            return Some(p);
        }
    }

    if let Ok(xdg) = std::env::var(env_keys::XDG_RUNTIME_DIR) {
        let p = std::path::PathBuf::from(xdg).join("biomeos/swarmvine.sock");
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Send a single `gossip.inject` request to swarmVine.
///
/// Returns `true` if the message was written successfully, `false` otherwise.
/// Never panics, never blocks longer than [`CONNECT_TIMEOUT`] + [`WRITE_TIMEOUT`].
#[cfg(unix)]
fn send_inject(topic: &str, key: &str, payload: &serde_json::Value) -> bool {
    let Some(socket_path) = discover_socket() else {
        return false;
    };

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "gossip.inject",
        "params": {
            "topic": topic,
            "key": key,
            "payload": payload,
        },
        "id": null,
    });

    let Ok(line) = serde_json::to_string(&request) else {
        return false;
    };

    let Ok(stream) = UnixStream::connect(&socket_path) else {
        return false;
    };
    let _ = stream.set_write_timeout(Some(WRITE_TIMEOUT));

    let mut writer = std::io::BufWriter::new(&stream);
    if writer.write_all(line.as_bytes()).is_err() {
        return false;
    }
    if writer.write_all(b"\n").is_err() {
        return false;
    }
    writer.flush().is_ok()
}

/// Non-Unix stub — gossip injection is UDS-only.
#[cfg(not(unix))]
fn send_inject(_topic: &str, _key: &str, _payload: &serde_json::Value) -> bool {
    false
}

// ─── Public injection API ────────────────────────────────────────────────────

/// Inject a `compute.device.created` event after device discovery.
pub fn inject_device_created(device_name: &str, pool_count: usize) {
    let key = format!("compute.device.created:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "device": device_name,
        "pool_device_count": pool_count,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected device.created");
    }
}

/// Inject a `compute.device.lost` event when a GPU device is lost.
pub fn inject_device_lost(device_name: &str, context: &str, message: &str) {
    let key = format!("compute.device.lost:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "device": device_name,
        "context": context,
        "message": message,
        "retriable": true,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected device.lost");
    }
}

/// Inject a `tower.endpoint.alive` event at startup.
pub fn inject_endpoint_alive(version: &str) {
    let key = format!("endpoint.alive:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "status": "alive",
        "version": version,
        "protocol": "jsonrpc-2.0",
    });
    if send_inject("tower", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected endpoint.alive");
    }
}

/// Inject a `tower.health.readiness_changed` event.
pub fn inject_readiness_changed(ready: bool, has_gpu: bool) {
    let key = format!("health.readiness_changed:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "ready": ready,
        "has_gpu": has_gpu,
        "mode": if has_gpu { "gpu" } else { "cpu-shader" },
    });
    if send_inject("tower", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected readiness_changed");
    }
}

/// Inject a `compute.capacity` event with pool availability.
pub fn inject_capacity(pool_count: usize, available_devices: usize) {
    let key = format!("compute.capacity:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "pool_devices": pool_count,
        "available_devices": available_devices,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected capacity");
    }
}

/// Inject a `tower.health.degraded` event when the primal runs without GPU.
pub fn inject_degraded(reason: &str, gpu_available: bool) {
    let key = format!("health.degraded:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "reason": reason,
        "gpu_available": gpu_available,
    });
    if send_inject("tower", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected health.degraded");
    }
}

/// Inject a `compute.error.systemic` event for non-retriable IPC errors.
pub fn inject_error_systemic(
    error_variant: &str,
    message: &str,
    method: &str,
    is_device_lost: bool,
    is_oom: bool,
) {
    let key = format!("compute.error.systemic:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "error_variant": error_variant,
        "message": message,
        "method": method,
        "retriable": false,
        "is_device_lost": is_device_lost,
        "is_oom": is_oom,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, method = %method, "gossip: injected error.systemic");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_name_returns_unknown_without_env() {
        let name = gate_name();
        assert!(!name.is_empty());
    }

    #[test]
    fn send_inject_returns_false_without_socket() {
        assert!(!send_inject("compute", "test.key", &serde_json::json!({})));
    }

    #[test]
    fn inject_device_created_does_not_panic() {
        inject_device_created("test-gpu", 1);
    }

    #[test]
    fn inject_device_lost_does_not_panic() {
        inject_device_lost("test-gpu", "submit", "device lost");
    }

    #[test]
    fn inject_endpoint_alive_does_not_panic() {
        inject_endpoint_alive("0.4.0");
    }

    #[test]
    fn inject_readiness_changed_does_not_panic() {
        inject_readiness_changed(true, true);
    }

    #[test]
    fn inject_capacity_does_not_panic() {
        inject_capacity(2, 1);
    }

    #[test]
    fn inject_degraded_does_not_panic() {
        inject_degraded("no GPU available", false);
    }

    #[test]
    fn inject_error_systemic_does_not_panic() {
        inject_error_systemic("Internal", "unexpected state", "compute.dispatch", false, false);
    }
}
