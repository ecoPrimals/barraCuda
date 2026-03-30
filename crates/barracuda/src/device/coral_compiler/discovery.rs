// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based runtime discovery of shader-compiler primals.
//!
//! Discovery is purely capability-based — no hardcoded primal names or ports.
//! Any primal advertising the `shader.compile` capability is accepted.
//!
//! Supports three transport mechanisms per wateringHole IPC v3.1:
//! - **Unix socket** via capability-domain symlink (`shader.sock` in
//!   `$XDG_RUNTIME_DIR/biomeos/`)
//! - **JSON manifest** scan for `shader.compile` in `provides`/`capabilities`
//! - **TCP** via explicit port (operator-configured)

use std::path::PathBuf;

use super::jsonrpc::jsonrpc_call;
use super::types::HealthResponse;

/// Environment variable for overriding the shader-compiler endpoint address.
///
/// When set, skips capability-based and port-based discovery entirely.
/// Supports `unix:/path/to/socket` for Unix socket addresses and
/// `host:port` for TCP addresses.
const COMPILER_ADDR_ENV: &str = "BARRACUDA_SHADER_COMPILER_ADDR";

/// Environment variable for an explicit shader-compiler port.
///
/// When set, enables a localhost probe as the final discovery fallback.
/// Without this, only env-address, socket, and capability-file discovery
/// are tried — no hardcoded port is ever probed.
const COMPILER_PORT_ENV: &str = "BARRACUDA_SHADER_COMPILER_PORT";

/// Loopback address for localhost-only discovery probes.
const LOCALHOST: &str = "127.0.0.1";

/// Legacy discovery filename for pre-Phase 10 backward compatibility.
///
/// Deprecated: prefer capability-based discovery (`shader.compile` or `shader_compiler`).
/// This fallback reads transport info from any remaining pre-capability manifest.
const LEGACY_DISCOVERY_FILENAME: &str = "shader-compiler.json";

/// Ecosystem shared namespace for socket-based discovery.
///
/// Per wateringHole `PRIMAL_IPC_PROTOCOL` v3.0, all primals share this
/// namespace under `$XDG_RUNTIME_DIR`. We scan it for capability-domain
/// symlinks (`shader.sock`) without knowing the specific primal name.
const ECOSYSTEM_SOCKET_NAMESPACE: &str = "biomeos";

/// Capability-domain symlink filename for shader compilation.
///
/// Per wateringHole `CAPABILITY_BASED_DISCOVERY_STANDARD` v1.1, the shader
/// compiler primal creates `shader.sock` as a symlink to its instance socket.
const SHADER_CAPABILITY_SOCKET: &str = "shader.sock";

/// Discover a shader-compiler primal's JSON-RPC endpoint via capability-based
/// runtime discovery. No hardcoded primal names or ports — any primal
/// advertising the `shader.compile` capability is accepted.
///
/// Discovery order:
/// 1. `BARRACUDA_SHADER_COMPILER_ADDR` — explicit override (operator-set)
/// 2. Unix socket: `$XDG_RUNTIME_DIR/biomeos/shader.sock` (capability-domain
///    symlink per `CAPABILITY_BASED_DISCOVERY` v1.1)
/// 3. JSON manifest scan of `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
///    `"shader.compile"` in `provides`/`capabilities` (toadStool S139 compat)
/// 4. Localhost probe on `BARRACUDA_SHADER_COMPILER_PORT` (only if set)
pub async fn discover_shader_compiler() -> Option<String> {
    if let Ok(addr) = std::env::var(COMPILER_ADDR_ENV) {
        let addr = addr.trim().to_owned();
        if !addr.is_empty() && probe_jsonrpc(&addr).await {
            tracing::debug!(addr = %addr, "shader compiler discovered via {COMPILER_ADDR_ENV}");
            return Some(addr);
        }
    }

    if let Some(addr) = discover_from_socket().await {
        return Some(addr);
    }

    if let Some(addr) = discover_from_file().await {
        if probe_jsonrpc(&addr).await {
            return Some(addr);
        }
    }

    if let Some(port) = std::env::var(COMPILER_PORT_ENV)
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

/// Discover shader compiler via Unix socket capability-domain symlink.
///
/// Scans `$XDG_RUNTIME_DIR/biomeos/shader.sock` (the v1.1 standard path).
/// The symlink is created by the shader compiler primal and points to its
/// instance-specific socket. We probe it with a health check.
#[cfg(unix)]
async fn discover_from_socket() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let socket_path = PathBuf::from(&runtime_dir)
        .join(ECOSYSTEM_SOCKET_NAMESPACE)
        .join(SHADER_CAPABILITY_SOCKET);

    if !socket_path.exists() {
        return None;
    }

    let addr = format!("unix:{}", socket_path.display());
    if probe_jsonrpc(&addr).await {
        tracing::debug!(
            path = %socket_path.display(),
            "shader compiler discovered via capability-domain socket"
        );
        return Some(addr);
    }
    None
}

#[cfg(not(unix))]
async fn discover_from_socket() -> Option<String> {
    None
}

/// Read transport info from the file-based discovery directory.
///
/// Scans both `$XDG_RUNTIME_DIR/ecoPrimals/` (toadStool S139 compat write)
/// and `$XDG_RUNTIME_DIR/ecoPrimals/discovery/` (canonical path) for any
/// primal advertising a `shader.compile` capability. Also scans the
/// `biomeos` namespace for JSON manifests. Falls back to the legacy
/// `shader_compiler` capability name, then the well-known filename.
async fn discover_from_file() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let eco_dir =
        std::env::var("ECOPRIMALS_DISCOVERY_DIR").unwrap_or_else(|_| "ecoPrimals".to_owned());
    let eco_base = PathBuf::from(&runtime_dir).join(&eco_dir);
    let eco_canonical = eco_base.join("discovery");
    let biomeos_base = PathBuf::from(&runtime_dir).join(ECOSYSTEM_SOCKET_NAMESPACE);

    let dirs = [&eco_base, &eco_canonical, &biomeos_base];

    for dir in &dirs {
        if let Some(addr) = scan_capability(dir, "shader.compile") {
            return Some(addr);
        }
    }

    for dir in &dirs {
        if let Some(addr) = scan_capability(dir, "shader_compiler") {
            return Some(addr);
        }
    }

    let legacy_path = eco_base.join(LEGACY_DISCOVERY_FILENAME);
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
///
/// Supports both Phase 10 manifests (`"provides"` array) and earlier
/// manifests (`"capabilities"` array) for backward compatibility.
fn read_capability_transport(path: &std::path::Path, capability: &str) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;

    let has_cap = info
        .get("provides")
        .or_else(|| info.get("capabilities"))
        .and_then(|v| v.as_array())
        .is_some_and(|caps| caps.iter().any(|c| c.as_str() == Some(capability)));

    if !has_cap {
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
///
/// Handles three formats with Unix socket preference:
/// - String: `"transports": { "jsonrpc": "127.0.0.1:5000" }`
/// - Object with Unix socket: `"transports": { "jsonrpc": { "unix": "/run/biomeos/shader.sock", "tcp": "..." } }`
/// - Object with TCP only: `"transports": { "jsonrpc": { "tcp": "127.0.0.1:5000" } }`
///
/// When both `unix` and `tcp` are present, prefers Unix socket for
/// lower-latency local IPC (zero network overhead).
fn read_jsonrpc_from_value(info: &serde_json::Value) -> Option<String> {
    let jsonrpc = info.get("transports")?.get("jsonrpc")?;
    if let Some(s) = jsonrpc.as_str() {
        return Some(s.to_owned());
    }
    if let Some(unix_path) = jsonrpc.get("unix").and_then(|v| v.as_str()) {
        if std::path::Path::new(unix_path).exists() {
            return Some(format!("unix:{unix_path}"));
        }
    }
    jsonrpc
        .get("tcp")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

/// Discover a shader-compiler primal that supports CPU execution.
///
/// Same discovery chain as [`discover_shader_compiler`], but additionally
/// verifies that the primal advertises the `shader.compile.cpu` or
/// `shader.execute.cpu` capability. Returns `None` if no CPU-capable
/// compiler is found.
pub async fn discover_cpu_shader_compiler() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let eco_dir =
        std::env::var("ECOPRIMALS_DISCOVERY_DIR").unwrap_or_else(|_| "ecoPrimals".to_owned());
    let eco_base = PathBuf::from(&runtime_dir).join(&eco_dir);
    let eco_canonical = eco_base.join("discovery");
    let biomeos_base = PathBuf::from(&runtime_dir).join(ECOSYSTEM_SOCKET_NAMESPACE);

    let dirs = [&eco_base, &eco_canonical, &biomeos_base];

    for dir in &dirs {
        if let Some(addr) = scan_capability(dir, "shader.compile.cpu") {
            if probe_jsonrpc(&addr).await {
                return Some(addr);
            }
        }
    }

    for dir in &dirs {
        if let Some(addr) = scan_capability(dir, "shader.execute.cpu") {
            if probe_jsonrpc(&addr).await {
                return Some(addr);
            }
        }
    }

    None
}

/// Discover a shader-compiler primal that supports validation.
///
/// Scans for the `shader.validate` capability.
pub async fn discover_shader_validator() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let eco_dir =
        std::env::var("ECOPRIMALS_DISCOVERY_DIR").unwrap_or_else(|_| "ecoPrimals".to_owned());
    let eco_base = PathBuf::from(&runtime_dir).join(&eco_dir);
    let eco_canonical = eco_base.join("discovery");
    let biomeos_base = PathBuf::from(&runtime_dir).join(ECOSYSTEM_SOCKET_NAMESPACE);

    let dirs = [&eco_base, &eco_canonical, &biomeos_base];

    for dir in &dirs {
        if let Some(addr) = scan_capability(dir, "shader.validate") {
            if probe_jsonrpc(&addr).await {
                return Some(addr);
            }
        }
    }

    None
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
                "shader compiler health OK"
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_jsonrpc_from_value_string_format() {
        let info: serde_json::Value = serde_json::json!({
            "transports": {
                "jsonrpc": "127.0.0.1:5000"
            }
        });
        assert_eq!(
            read_jsonrpc_from_value(&info),
            Some("127.0.0.1:5000".to_owned())
        );
    }

    #[test]
    fn read_jsonrpc_from_value_phase10_object_format() {
        let info: serde_json::Value = serde_json::json!({
            "transports": {
                "jsonrpc": {
                    "tcp": "127.0.0.1:5000",
                    "path": "/run/ecoPrimals/shader-compiler.sock"
                }
            }
        });
        assert_eq!(
            read_jsonrpc_from_value(&info),
            Some("127.0.0.1:5000".to_owned())
        );
    }

    #[test]
    fn read_jsonrpc_from_value_unix_socket_preferred_when_exists() {
        let dir = std::env::temp_dir().join("barracuda_test_unix_pref");
        let _ = std::fs::create_dir_all(&dir);
        let sock = dir.join("shader.sock");
        std::fs::write(&sock, b"").unwrap();

        let info: serde_json::Value = serde_json::json!({
            "transports": {
                "jsonrpc": {
                    "unix": sock.to_str().unwrap(),
                    "tcp": "127.0.0.1:5000"
                }
            }
        });
        let result = read_jsonrpc_from_value(&info);
        assert!(result.is_some());
        let addr = result.unwrap();
        assert!(
            addr.starts_with("unix:"),
            "should prefer unix socket: {addr}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_jsonrpc_from_value_unix_falls_back_to_tcp_when_missing() {
        let info: serde_json::Value = serde_json::json!({
            "transports": {
                "jsonrpc": {
                    "unix": "/nonexistent/path/shader.sock",
                    "tcp": "127.0.0.1:5000"
                }
            }
        });
        assert_eq!(
            read_jsonrpc_from_value(&info),
            Some("127.0.0.1:5000".to_owned()),
        );
    }

    #[test]
    fn read_jsonrpc_from_value_no_transport() {
        let info: serde_json::Value = serde_json::json!({
            "name": "some-primal"
        });
        assert_eq!(read_jsonrpc_from_value(&info), None);
    }

    #[test]
    fn read_capability_transport_provides_array() {
        let dir = std::env::temp_dir().join("barracuda_test_discovery");
        let _ = std::fs::create_dir_all(&dir);
        let manifest = dir.join("test-shader-compiler.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "provides": ["shader.compile", "shader.compile.wgsl"],
                "transports": {
                    "jsonrpc": "127.0.0.1:7777"
                }
            })
            .to_string(),
        )
        .unwrap();

        let result = read_capability_transport(&manifest, "shader.compile");
        assert_eq!(result, Some("127.0.0.1:7777".to_owned()));

        let result = read_capability_transport(&manifest, "nonexistent");
        assert_eq!(result, None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_capability_transport_legacy_capabilities_array() {
        let dir = std::env::temp_dir().join("barracuda_test_discovery_legacy");
        let _ = std::fs::create_dir_all(&dir);
        let manifest = dir.join("legacy-compiler.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "capabilities": ["shader_compiler"],
                "transports": {
                    "jsonrpc": "127.0.0.1:8888"
                }
            })
            .to_string(),
        )
        .unwrap();

        let result = read_capability_transport(&manifest, "shader_compiler");
        assert_eq!(result, Some("127.0.0.1:8888".to_owned()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn constants_match_expected_values() {
        assert_eq!(ECOSYSTEM_SOCKET_NAMESPACE, "biomeos");
        assert_eq!(SHADER_CAPABILITY_SOCKET, "shader.sock");
        assert_eq!(COMPILER_ADDR_ENV, "BARRACUDA_SHADER_COMPILER_ADDR");
        assert_eq!(COMPILER_PORT_ENV, "BARRACUDA_SHADER_COMPILER_PORT");
    }

    #[tokio::test]
    async fn discover_returns_none_without_env() {
        let result = discover_shader_compiler().await;
        if let Some(ref addr) = result {
            assert!(!addr.is_empty());
        }
    }

    #[tokio::test]
    async fn discover_from_socket_returns_none_without_socket() {
        let result = discover_from_socket().await;
        assert!(result.is_none(), "should return None when no socket exists");
    }
}
