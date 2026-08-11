// SPDX-License-Identifier: AGPL-3.0-or-later
//! Runtime gossip injection for the swarmVine epidemic mesh.
//!
//! Complements the startup gossip injections in `barracuda-core::ipc::gossip`
//! with runtime events from GPU dispatch, compilation, and error paths.
//! All calls are fire-and-forget: if swarmVine is absent, injections silently
//! return `false`. Zero-cost when the mesh is not running.

use std::io::Write;
#[cfg(unix)]
use std::os::unix::net::UnixStream;
use std::sync::OnceLock;
use std::time::Duration;

use crate::env_keys;

/// swarmVine socket environment variable override.
const SWARMVINE_SOCKET_ENV: &str = "SWARMVINE_SOCKET";

/// Local gate name for gossip key suffixes.
const GATE_NAME_ENV: &str = "GATE_NAME";

/// Write timeout for a single gossip injection.
const WRITE_TIMEOUT: Duration = Duration::from_millis(500);

fn gate_name() -> &'static str {
    static GATE: OnceLock<String> = OnceLock::new();
    GATE.get_or_init(|| std::env::var(GATE_NAME_ENV).unwrap_or_else(|_| "unknown".into()))
}

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

#[cfg(not(unix))]
fn send_inject(_topic: &str, _key: &str, _payload: &serde_json::Value) -> bool {
    false
}

/// Inject `compute.device.lost` when a GPU device is lost.
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

/// Inject `compute.device.oom` when a GPU driver reports out-of-memory.
pub fn inject_device_oom(
    device_name: &str,
    message: &str,
    pool_index: usize,
    migration_attempt: bool,
) {
    let key = format!("compute.device.oom:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "device": device_name,
        "message": message,
        "pool_index": pool_index,
        "migration_attempt": migration_attempt,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected device.oom");
    }
}

/// Inject `compute.device.tier_fallback` when device discovery falls back.
pub fn inject_tier_fallback(from_tier: &str, to_tier: &str, reason: &str) {
    let key = format!("compute.device.tier_fallback:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "from_tier": from_tier,
        "to_tier": to_tier,
        "reason": reason,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected tier_fallback");
    }
}

/// Inject `compute.shader.cache_hit` on sovereign or coral cache hit.
pub fn inject_shader_cache_hit(shader_hash: u64, cache_tier: &str) {
    let key = format!("compute.shader.cache_hit:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "shader_hash": shader_hash,
        "cache_tier": cache_tier,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected shader.cache_hit");
    }
}

/// Inject `compute.shader.cache_miss` when a shader requires live compilation.
pub fn inject_shader_cache_miss(shader_hash: u64) {
    let key = format!("compute.shader.cache_miss:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "shader_hash": shader_hash,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected shader.cache_miss");
    }
}

/// Inject `compute.shader.compile_success` after a successful compilation.
pub fn inject_compile_success(
    shader_hash: impl std::fmt::Display,
    target_arch: &str,
    binary_bytes: usize,
    compile_path: &str,
) {
    let key = format!("compute.shader.compile_success:{}:barracuda", gate_name());
    let hash_str = shader_hash.to_string();
    let payload = serde_json::json!({
        "shader_hash": hash_str,
        "target_arch": target_arch,
        "binary_bytes": binary_bytes,
        "compile_path": compile_path,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected shader.compile_success");
    }
}

/// Inject `compute.shader.compile_failure` when compilation fails.
pub fn inject_compile_failure(target_arch: &str, error_kind: &str, message: &str) {
    let key = format!("compute.shader.compile_failure:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "target_arch": target_arch,
        "error_kind": error_kind,
        "message": message,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected shader.compile_failure");
    }
}

/// Inject `compute.compiler.peer_status` when coral availability changes.
pub fn inject_compiler_peer_status(coral_available: bool, compiler_addr: Option<&str>) {
    let key = format!("compute.compiler.peer_status:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "coral_available": coral_available,
        "compiler_addr": compiler_addr,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected compiler.peer_status");
    }
}

/// Inject `compute.dispatch.stall` when GPU poll times out.
pub fn inject_dispatch_stall(timeout_secs: u64, context: &str) {
    let key = format!("compute.dispatch.stall:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "timeout_secs": timeout_secs,
        "context": context,
        "retriable": true,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected dispatch.stall");
    }
}

/// Inject `compute.quota.exceeded` when resource allocation exceeds quota.
pub fn inject_quota_exceeded(
    requested_bytes: u64,
    limit_bytes: u64,
    device_name: &str,
    operation: &str,
) {
    let key = format!("compute.quota.exceeded:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "requested_bytes": requested_bytes,
        "limit_bytes": limit_bytes,
        "device_name": device_name,
        "operation": operation,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected quota.exceeded");
    }
}

/// Inject `compute.oom.migration` when OOM triggers workload migration.
pub fn inject_oom_migration(
    attempt: usize,
    from_device: &str,
    to_device: Option<&str>,
    excluded_count: usize,
) {
    let key = format!("compute.oom.migration:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "attempt": attempt,
        "from_device": from_device,
        "to_device": to_device,
        "excluded_count": excluded_count,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected oom.migration");
    }
}

/// Inject `compute.precision.route` when precision routing selects a tier.
pub fn inject_precision_route(domain: &str, tier: &str, dispatch_path: &str) {
    let key = format!("compute.precision.route:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "domain": domain,
        "tier": tier,
        "dispatch_path": dispatch_path,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected precision.route");
    }
}

/// Inject `compute.device.recovered` when a fresh device replaces a lost one.
pub fn inject_device_recovered(device_name: &str, recovery_context: &str) {
    let key = format!("compute.device.recovered:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "device": device_name,
        "recovery_context": recovery_context,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected device.recovered");
    }
}

/// Inject `compute.precision.tier_degraded` when hardware forces tier degradation.
pub fn inject_precision_tier_degraded(
    requested_tier: &str,
    actual_tier: &str,
    reason: &str,
    adapter: &str,
) {
    let key = format!("compute.precision.tier_degraded:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "requested_tier": requested_tier,
        "actual_tier": actual_tier,
        "reason": reason,
        "adapter": adapter,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected precision.tier_degraded");
    }
}

/// Inject `compute.error.retriable_exhausted` when all retries fail.
pub fn inject_retriable_exhausted(error_kind: &str, attempts: usize, last_message: &str) {
    let key = format!("compute.error.retriable_exhausted:{}:barracuda", gate_name());
    let payload = serde_json::json!({
        "error_kind": error_kind,
        "attempts": attempts,
        "last_message": last_message,
    });
    if send_inject("compute", &key, &payload) {
        tracing::debug!(key = %key, "gossip: injected error.retriable_exhausted");
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
    fn inject_device_lost_does_not_panic() {
        inject_device_lost("test-gpu", "submit", "device lost");
    }

    #[test]
    fn inject_device_oom_does_not_panic() {
        inject_device_oom("test-gpu", "out of memory", 0, false);
    }

    #[test]
    fn inject_tier_fallback_does_not_panic() {
        inject_tier_fallback("wgpu_gpu", "wgpu_cpu", "gpu unavailable");
    }

    #[test]
    fn inject_shader_cache_hit_does_not_panic() {
        inject_shader_cache_hit(0xdead_beef, "sovereign");
    }

    #[test]
    fn inject_shader_cache_miss_does_not_panic() {
        inject_shader_cache_miss(0xdead_beef);
    }

    #[test]
    fn inject_compile_success_does_not_panic() {
        inject_compile_success(0xdead_beef_u64, "spirv", 4096, "wgsl_direct");
    }

    #[test]
    fn inject_compile_failure_does_not_panic() {
        inject_compile_failure("spirv", "compile_failed", "naga validation error");
    }

    #[test]
    fn inject_compiler_peer_status_does_not_panic() {
        inject_compiler_peer_status(true, Some("unix:/tmp/coral.sock"));
        inject_compiler_peer_status(false, None);
    }

    #[test]
    fn inject_dispatch_stall_does_not_panic() {
        inject_dispatch_stall(30, "poll_safe");
    }

    #[test]
    fn inject_quota_exceeded_does_not_panic() {
        inject_quota_exceeded(1_073_741_824, 536_870_912, "test-gpu", "create_buffer");
    }

    #[test]
    fn inject_oom_migration_does_not_panic() {
        inject_oom_migration(1, "gpu-0", Some("gpu-1"), 1);
    }

    #[test]
    fn inject_precision_route_does_not_panic() {
        inject_precision_route("stats", "f64", "wgpu");
    }

    #[test]
    fn inject_device_recovered_does_not_panic() {
        inject_device_recovered("test-gpu", "pool_slot_replace");
    }

    #[test]
    fn inject_precision_tier_degraded_does_not_panic() {
        inject_precision_tier_degraded("F64", "DF64", "probe_failed", "llvmpipe");
    }

    #[test]
    fn inject_retriable_exhausted_does_not_panic() {
        inject_retriable_exhausted("oom_migration", 3, "all pool devices exhausted");
    }
}
