// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based runtime discovery of shader-compiler primals.

use std::path::PathBuf;

use super::jsonrpc::jsonrpc_call;
use super::types::HealthResponse;

/// Environment variable for overriding the shader-compiler endpoint address.
///
/// When set, skips capability-based and port-based discovery entirely.
const CORALREEF_ADDR_ENV: &str = "BARRACUDA_SHADER_COMPILER_ADDR";

/// Environment variable for an explicit shader-compiler port.
///
/// When set, enables a localhost probe as the final discovery fallback.
/// Without this, only env-address and capability-file discovery are tried —
/// no hardcoded port is ever probed.
const CORALREEF_PORT_ENV: &str = "BARRACUDA_SHADER_COMPILER_PORT";

/// Loopback address for localhost-only discovery probes.
const LOCALHOST: &str = "127.0.0.1";

/// Well-known discovery filename for pre-Phase 10 backward compatibility.
const LEGACY_DISCOVERY_FILENAME: &str = "coralreef-core.json";

/// Discover a shader-compiler primal's JSON-RPC endpoint via capability-based
/// runtime discovery. No hardcoded primal names or ports — any primal
/// advertising the `shader.compile` capability is accepted.
///
/// Discovery order:
/// 1. `BARRACUDA_SHADER_COMPILER_ADDR` — explicit override (operator-set)
/// 2. Capability scan of `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
///    `"shader.compile"` in the `capabilities` array (falls back to legacy
///    `"shader_compiler"` for pre-Phase 10 primals)
/// 3. Localhost probe on `BARRACUDA_SHADER_COMPILER_PORT` (only if set)
pub async fn discover_coralreef() -> Option<String> {
    if let Ok(addr) = std::env::var(CORALREEF_ADDR_ENV) {
        let addr = addr.trim().to_owned();
        if !addr.is_empty() && probe_jsonrpc(&addr).await {
            tracing::debug!(addr = %addr, "shader compiler discovered via {CORALREEF_ADDR_ENV}");
            return Some(addr);
        }
    }

    if let Some(addr) = discover_from_file().await {
        if probe_jsonrpc(&addr).await {
            return Some(addr);
        }
    }

    if let Some(port) = std::env::var(CORALREEF_PORT_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<u16>().ok())
    {
        let explicit_addr = format!("{LOCALHOST}:{port}");
        if probe_jsonrpc(&explicit_addr).await {
            return Some(explicit_addr);
        }
    }

    None
}

/// Read transport info from the file-based discovery directory.
///
/// Scans `$XDG_RUNTIME_DIR/ecoPrimals/` for any primal advertising a
/// `shader.compile` capability via its transport manifest. Falls back to
/// the legacy `shader_compiler` capability name, then the well-known
/// `coralreef-core.json` filename for backward compat.
async fn discover_from_file() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let discovery_dir = PathBuf::from(runtime_dir).join("ecoPrimals");

    // Phase 10 semantic capability name
    if let Some(addr) = scan_capability(&discovery_dir, "shader.compile") {
        return Some(addr);
    }

    // Pre-Phase 10 capability name
    if let Some(addr) = scan_capability(&discovery_dir, "shader_compiler") {
        return Some(addr);
    }

    // Fallback: well-known filename for backward compatibility
    let legacy_path = discovery_dir.join(LEGACY_DISCOVERY_FILENAME);
    read_jsonrpc_transport(&legacy_path)
}

/// Scan the discovery directory for any primal advertising a given capability.
fn scan_capability(dir: &std::path::Path, capability: &str) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            if let Some(addr) = read_capability_transport(&path, capability) {
                return Some(addr);
            }
        }
    }
    None
}

/// Read a primal's transport manifest and check for a specific capability.
fn read_capability_transport(path: &std::path::Path, capability: &str) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;
    let caps = info.get("capabilities")?.as_array()?;
    if !caps.iter().any(|c| c.as_str() == Some(capability)) {
        return None;
    }
    read_jsonrpc_from_value(&info)
}

/// Extract the JSON-RPC transport address from a primal manifest.
fn read_jsonrpc_transport(path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;
    read_jsonrpc_from_value(&info)
}

/// Extract JSON-RPC address from a parsed transport manifest.
fn read_jsonrpc_from_value(info: &serde_json::Value) -> Option<String> {
    info.get("transports")
        .and_then(|t| t.get("jsonrpc"))
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

/// Probe whether a JSON-RPC endpoint is alive via `shader.compile.status`.
///
/// Falls back to the legacy `compiler.health` method for pre-Phase 10
/// coralReef instances.
pub async fn probe_jsonrpc(addr: &str) -> bool {
    match jsonrpc_call::<(), HealthResponse>(addr, "shader.compile.status", &()).await {
        Ok(resp) => {
            tracing::debug!(
                name = resp.name,
                version = resp.version,
                "coralReef health OK"
            );
            true
        }
        Err(_) => {
            // Backward compat: try pre-Phase 10 method name
            jsonrpc_call::<(), HealthResponse>(addr, "compiler.health", &())
                .await
                .is_ok()
        }
    }
}
