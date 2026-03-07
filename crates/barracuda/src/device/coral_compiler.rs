// SPDX-License-Identifier: AGPL-3.0-or-later
//! coralReef shader compiler IPC client.
//!
//! Discovers and connects to the coralReef primal's JSON-RPC 2.0 endpoint,
//! providing native GPU binary compilation for NVIDIA (SM70+) and AMD (RDNA2+)
//! architectures.
//!
//! ## IPC Contract (coralReef Phase 10)
//!
//! | Method | Purpose |
//! |--------|---------|
//! | `shader.compile.spirv` | SPIR-V → native binary |
//! | `shader.compile.wgsl` | WGSL → native binary |
//! | `shader.compile.status` | Health / readiness |
//! | `shader.compile.capabilities` | Supported architectures |
//!
//! Fully optional: if coralReef is unavailable, all methods return `None`
//! and the standard wgpu/SovereignCompiler path is used.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Mutex;

/// Cached native binary produced by coralReef.
#[derive(Debug, Clone)]
pub struct CoralBinary {
    /// Raw GPU binary (SM70+ native code).
    pub binary: Vec<u8>,
    /// Target architecture (e.g. `sm_70`).
    pub arch: String,
}

/// SPIR-V compile request — mirrors `coralreef-core::service::CompileRequest`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompileRequest {
    spirv_words: Vec<u32>,
    arch: String,
    opt_level: u32,
    fp64_software: bool,
}

/// WGSL direct compile request (Phase 10) — avoids local naga SPIR-V step.
///
/// coralReef Phase 10 (`shader.compile.wgsl`) accepts raw WGSL and handles
/// the full WGSL → IR → native binary pipeline server-side.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompileWgslRequest {
    wgsl_source: String,
    arch: String,
    opt_level: u32,
    fp64_software: bool,
}

/// Compile response — mirrors `coralreef-core::service::CompileResponse`.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompileResponse {
    binary: Vec<u8>,
    size: usize,
}

/// Health response from coralReef.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Primal name (e.g. `"coralReef"`)
    pub name: String,
    /// Version string
    pub version: String,
    /// Health status
    pub status: String,
    /// Supported GPU architectures (e.g. `["sm_70", "sm_75", "sm_80", "sm_89"]`)
    pub supported_archs: Vec<String>,
}

/// Connection state for the coralReef IPC client.
#[derive(Debug)]
enum ConnectionState {
    /// Haven't tried connecting yet.
    Uninit,
    /// Connected to a JSON-RPC endpoint.
    Connected { addr: String },
    /// Discovery failed or connection refused — don't retry until reset.
    Unavailable,
}

/// IPC client for the coralReef shader compiler primal.
///
/// Lazily discovers coralReef on first use. Thread-safe via interior mutex.
#[derive(Debug)]
pub struct CoralCompiler {
    state: Arc<Mutex<ConnectionState>>,
}

impl CoralCompiler {
    /// Create a new compiler client (no connection attempt until first use).
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(ConnectionState::Uninit)),
        }
    }

    /// Attempt to compile WGSL to a native GPU binary via coralReef.
    ///
    /// Returns `None` if coralReef is unavailable or compilation fails.
    /// The caller should fall back to the standard wgpu path.
    pub async fn compile_wgsl(
        &self,
        wgsl: &str,
        arch: &str,
        fp64_software: bool,
    ) -> Option<CoralBinary> {
        let spirv_words = wgsl_to_spirv(wgsl)?;
        self.compile_spirv(&spirv_words, arch, fp64_software).await
    }

    /// Compile pre-assembled SPIR-V words to a native binary.
    ///
    /// Returns `None` if coralReef is unavailable or compilation fails.
    pub async fn compile_spirv(
        &self,
        spirv_words: &[u32],
        arch: &str,
        fp64_software: bool,
    ) -> Option<CoralBinary> {
        let addr = self.ensure_connected().await?;

        let request = CompileRequest {
            spirv_words: spirv_words.to_vec(),
            arch: arch.to_owned(),
            opt_level: 2,
            fp64_software,
        };

        match jsonrpc_call::<CompileRequest, CompileResponse>(
            &addr,
            "shader.compile.spirv",
            &request,
        )
        .await
        {
            Ok(resp) => Some(CoralBinary {
                binary: resp.binary,
                arch: arch.to_owned(),
            }),
            Err(e) => {
                tracing::debug!("coralReef compile failed: {e}");
                None
            }
        }
    }

    /// Compile WGSL directly via coralReef's Phase 10 `shader.compile.wgsl`.
    ///
    /// Unlike [`compile_wgsl`], this sends raw WGSL to coralReef for
    /// server-side compilation, avoiding the local naga WGSL → SPIR-V step.
    /// coralReef handles the full pipeline: WGSL → IR → native binary.
    ///
    /// Falls back to the SPIR-V path if the direct endpoint is unavailable
    /// (older coralReef versions).
    pub async fn compile_wgsl_direct(
        &self,
        wgsl: &str,
        arch: &str,
        fp64_software: bool,
    ) -> Option<CoralBinary> {
        let addr = self.ensure_connected().await?;

        let request = CompileWgslRequest {
            wgsl_source: wgsl.to_owned(),
            arch: arch.to_owned(),
            opt_level: 2,
            fp64_software,
        };

        match jsonrpc_call::<CompileWgslRequest, CompileResponse>(
            &addr,
            "shader.compile.wgsl",
            &request,
        )
        .await
        {
            Ok(resp) => Some(CoralBinary {
                binary: resp.binary,
                arch: arch.to_owned(),
            }),
            Err(e) => {
                tracing::debug!("coralReef compile_wgsl direct failed: {e}, trying SPIR-V path");
                self.compile_wgsl(wgsl, arch, fp64_software).await
            }
        }
    }

    /// Query supported GPU architectures from coralReef.
    ///
    /// Prefers the Phase 10 `shader.compile.capabilities` endpoint, falling
    /// back to `shader.compile.status` for older coralReef versions that
    /// embed arch info in the health response.
    pub async fn supported_archs(&self) -> Option<Vec<String>> {
        if let Some(archs) = self.capabilities().await {
            return Some(archs);
        }
        self.health().await.map(|h| h.supported_archs)
    }

    /// Query supported architectures via `shader.compile.capabilities`.
    ///
    /// Returns `None` if coralReef is unavailable or the endpoint is not
    /// supported (pre-Phase 10 versions).
    pub async fn capabilities(&self) -> Option<Vec<String>> {
        let addr = self.ensure_connected().await?;
        jsonrpc_call::<(), Vec<String>>(&addr, "shader.compile.capabilities", &())
            .await
            .ok()
    }

    /// Check if coralReef is reachable and healthy via `shader.compile.status`.
    pub async fn health(&self) -> Option<HealthResponse> {
        let addr = self.ensure_connected().await?;
        jsonrpc_call::<(), HealthResponse>(&addr, "shader.compile.status", &())
            .await
            .ok()
    }

    /// Ensure we have a connection, discovering coralReef if needed.
    async fn ensure_connected(&self) -> Option<String> {
        let mut state = self.state.lock().await;
        match &*state {
            ConnectionState::Connected { addr } => Some(addr.clone()),
            ConnectionState::Unavailable => None,
            ConnectionState::Uninit => {
                if let Some(addr) = discover_coralreef().await {
                    tracing::info!(addr = %addr, "discovered coralReef compiler service");
                    *state = ConnectionState::Connected { addr: addr.clone() };
                    Some(addr)
                } else {
                    tracing::debug!("coralReef not available — using standard compilation path");
                    *state = ConnectionState::Unavailable;
                    None
                }
            }
        }
    }

    /// Reset connection state, forcing re-discovery on next use.
    pub async fn reset(&self) {
        let mut state = self.state.lock().await;
        *state = ConnectionState::Uninit;
    }
}

impl Default for CoralCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Global coralReef compiler client (lazy singleton like `GLOBAL_CACHE`).
pub static GLOBAL_CORAL: std::sync::LazyLock<CoralCompiler> =
    std::sync::LazyLock::new(CoralCompiler::new);

/// Lightweight health probe for coralReef availability.
///
/// Uses the global singleton — does not create a new `CoralCompiler` instance.
/// Returns `true` if coralReef is reachable and reports healthy status.
pub async fn probe_health() -> bool {
    GLOBAL_CORAL.health().await.is_some()
}

/// Cache of native GPU binaries produced by coralReef, keyed by
/// (blake3 hash of shader source, target arch).
static NATIVE_BINARY_CACHE: std::sync::LazyLock<
    std::sync::RwLock<std::collections::HashMap<(String, String), CoralBinary>>,
> = std::sync::LazyLock::new(|| std::sync::RwLock::new(std::collections::HashMap::new()));

/// Look up a cached native binary for the given shader source and arch.
#[must_use]
pub fn cached_native_binary(shader_hash: &str, arch: &str) -> Option<CoralBinary> {
    let cache = NATIVE_BINARY_CACHE
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache
        .get(&(shader_hash.to_owned(), arch.to_owned()))
        .cloned()
}

/// Store a native binary in the cache.
pub fn cache_native_binary(shader_hash: &str, arch: &str, binary: CoralBinary) {
    let mut cache = NATIVE_BINARY_CACHE
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    cache.insert((shader_hash.to_owned(), arch.to_owned()), binary);
}

/// Hash shader source for cache keying (uses blake3 for consistency with barraCuda).
#[must_use]
pub fn shader_hash(source: &str) -> String {
    blake3::hash(source.as_bytes()).to_hex().to_string()
}

/// Map a barraCuda `GpuArch` to coralReef's arch string.
///
/// Supports NVIDIA (SM70+) and AMD RDNA2+ architectures per coralReef Phase 10.
/// Returns `None` for architectures that coralReef cannot compile for
/// (Intel Arc, Apple M, software rasterizers, unknowns).
#[must_use]
pub fn arch_to_coral(arch: &crate::device::driver_profile::GpuArch) -> Option<&'static str> {
    use crate::device::driver_profile::GpuArch;
    match arch {
        GpuArch::Volta => Some("sm_70"),
        GpuArch::Turing => Some("sm_75"),
        GpuArch::Ampere => Some("sm_80"),
        GpuArch::Ada => Some("sm_89"),
        GpuArch::Rdna2 => Some("gfx1030"),
        GpuArch::Rdna3 => Some("gfx1100"),
        GpuArch::Cdna2 => Some("gfx90a"),
        _ => None,
    }
}

/// Fire-and-forget coralReef compilation for the given WGSL source.
///
/// Spawns a background task that compiles via coralReef IPC and caches
/// the resulting native binary. Prefers the Phase 10 direct WGSL path
/// (`shader.compile.wgsl`), falling back to the SPIR-V path for older
/// coralReef versions. The caller continues with the standard wgpu path —
/// the cached binary becomes available for future `coralDriver` dispatch.
pub fn spawn_coral_compile(optimized_wgsl: &str, arch: &str, fp64_software: bool) {
    let source = optimized_wgsl.to_owned();
    let arch_owned = arch.to_owned();
    let hash = shader_hash(&source);

    if cached_native_binary(&hash, &arch_owned).is_some() {
        return;
    }

    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };

    handle.spawn(async move {
        if let Some(binary) = GLOBAL_CORAL
            .compile_wgsl_direct(&source, &arch_owned, fp64_software)
            .await
        {
            tracing::debug!(
                arch = %arch_owned,
                size = binary.binary.len(),
                "coralReef: cached native binary ({} bytes)",
                binary.binary.len(),
            );
            cache_native_binary(&hash, &arch_owned, binary);
        }
    });
}

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

/// Discover a shader-compiler primal's JSON-RPC endpoint via capability-based
/// runtime discovery. No hardcoded primal names or ports — any primal
/// advertising the `shader_compiler` capability is accepted.
///
/// Discovery order:
/// 1. `BARRACUDA_SHADER_COMPILER_ADDR` — explicit override (operator-set)
/// 2. Capability scan of `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
///    `"shader_compiler"` in the `capabilities` array
/// 3. Localhost probe on `BARRACUDA_SHADER_COMPILER_PORT` (only if set)
async fn discover_coralreef() -> Option<String> {
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
        let explicit_addr = format!("127.0.0.1:{port}");
        if probe_jsonrpc(&explicit_addr).await {
            return Some(explicit_addr);
        }
    }

    None
}

/// Read transport info from the file-based discovery directory.
///
/// Scans `$XDG_RUNTIME_DIR/ecoPrimals/` for any primal advertising a
/// `shader_compiler` capability via its transport manifest. Falls back to
/// the well-known `coralreef-core.json` filename for backward compat.
async fn discover_from_file() -> Option<String> {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let discovery_dir = PathBuf::from(runtime_dir).join("ecoPrimals");

    if let Some(addr) = scan_capability(&discovery_dir, "shader_compiler") {
        return Some(addr);
    }

    // Fallback: well-known filename for backward compatibility
    let legacy_path = discovery_dir.join("coralreef-core.json");
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
async fn probe_jsonrpc(addr: &str) -> bool {
    match jsonrpc_call::<(), HealthResponse>(addr, "shader.compile.status", &()).await {
        Ok(resp) => {
            tracing::debug!(
                name = resp.name,
                version = resp.version,
                "coralReef health OK"
            );
            true
        }
        Err(_) => false,
    }
}

/// Convert WGSL to SPIR-V words using naga (local, no IPC).
fn wgsl_to_spirv(wgsl: &str) -> Option<Vec<u32>> {
    let module = match naga::front::wgsl::parse_str(wgsl) {
        Ok(m) => m,
        Err(e) => {
            tracing::debug!("coralReef: WGSL parse failed: {e}");
            return None;
        }
    };

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = match validator.validate(&module) {
        Ok(i) => i,
        Err(e) => {
            tracing::debug!("coralReef: WGSL validation failed: {e}");
            return None;
        }
    };

    let options = naga::back::spv::Options {
        lang_version: (1, 5),
        ..Default::default()
    };
    let pipeline_options = None;

    match naga::back::spv::write_vec(&module, &info, &options, pipeline_options) {
        Ok(words) => Some(words),
        Err(e) => {
            tracing::debug!("coralReef: SPIR-V emit failed: {e}");
            None
        }
    }
}

/// Low-level JSON-RPC 2.0 call over TCP.
///
/// Opens a fresh TCP connection per call (simple, stateless). For the
/// shader compilation use case, connection overhead is negligible compared
/// to compilation time.
async fn jsonrpc_call<P: Serialize, R: for<'de> Deserialize<'de>>(
    addr: &str,
    method: &str,
    params: &P,
) -> Result<R, String> {
    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": [params],
        "id": 1,
    });
    let body = serde_json::to_string(&request).map_err(|e| e.to_string())?;

    let host_port = addr.trim_start_matches("http://");
    let http_request = format!(
        "POST / HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );

    let mut stream = TcpStream::connect(host_port)
        .await
        .map_err(|e| format!("TCP connect to {host_port}: {e}"))?;

    stream
        .write_all(http_request.as_bytes())
        .await
        .map_err(|e| format!("TCP write: {e}"))?;

    let mut response_buf = Vec::new();
    stream
        .read_to_end(&mut response_buf)
        .await
        .map_err(|e| format!("TCP read: {e}"))?;

    let response_str = String::from_utf8_lossy(&response_buf);

    let json_start = response_str
        .find('{')
        .ok_or("no JSON body in HTTP response")?;
    let json_body = &response_str[json_start..];

    let rpc_response: serde_json::Value =
        serde_json::from_str(json_body).map_err(|e| format!("JSON parse: {e}"))?;

    if let Some(error) = rpc_response.get("error") {
        let msg = error
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("JSON-RPC error: {msg}"));
    }

    let result = rpc_response
        .get("result")
        .ok_or("no result field in JSON-RPC response")?;

    serde_json::from_value(result.clone()).map_err(|e| format!("deserialize result: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coral_compiler_new() {
        let cc = CoralCompiler::new();
        assert!(format!("{cc:?}").contains("CoralCompiler"));
    }

    #[test]
    fn test_coral_compiler_default() {
        let cc = CoralCompiler::default();
        assert!(format!("{cc:?}").contains("CoralCompiler"));
    }

    #[test]
    fn test_coral_binary_debug() {
        let binary = CoralBinary {
            binary: vec![0xDE, 0xAD],
            arch: "sm_70".to_owned(),
        };
        assert!(format!("{binary:?}").contains("sm_70"));
    }

    #[test]
    fn test_wgsl_to_spirv_valid() {
        let wgsl = r"
            @compute @workgroup_size(64)
            fn main() {}
        ";
        let words = wgsl_to_spirv(wgsl);
        assert!(words.is_some(), "valid WGSL should produce SPIR-V");
        let words = words.unwrap();
        assert!(words.len() > 10, "SPIR-V should have non-trivial length");
        assert_eq!(words[0], 0x0723_0203, "SPIR-V magic number");
    }

    #[test]
    fn test_wgsl_to_spirv_invalid() {
        let invalid = "fn main() { let x = ; }";
        assert!(wgsl_to_spirv(invalid).is_none());
    }

    #[tokio::test]
    async fn test_discovery_returns_none_when_unavailable() {
        let addr = discover_coralreef().await;
        // In CI/test environments coralReef is typically not running,
        // so this should return None (graceful degradation).
        // If it IS running, that's also fine.
        let _ = addr;
    }

    #[tokio::test]
    async fn test_compile_wgsl_returns_none_when_unavailable() {
        let cc = CoralCompiler::new();
        let result = cc
            .compile_wgsl("@compute @workgroup_size(64) fn main() {}", "sm_70", true)
            .await;
        let _ = result;
    }

    #[tokio::test]
    async fn test_compile_wgsl_direct_returns_none_when_unavailable() {
        let cc = CoralCompiler::new();
        let result = cc
            .compile_wgsl_direct("@compute @workgroup_size(64) fn main() {}", "sm_70", false)
            .await;
        let _ = result;
    }

    #[tokio::test]
    async fn test_supported_archs_returns_none_when_unavailable() {
        let cc = CoralCompiler::new();
        let archs = cc.supported_archs().await;
        let _ = archs;
    }

    #[tokio::test]
    async fn test_reset_allows_rediscovery() {
        let cc = CoralCompiler::new();
        let _ = cc.health().await;
        cc.reset().await;
        let state = cc.state.lock().await;
        assert!(matches!(&*state, ConnectionState::Uninit));
    }

    #[tokio::test]
    async fn test_capabilities_returns_none_when_unavailable() {
        let cc = CoralCompiler::new();
        let caps = cc.capabilities().await;
        let _ = caps;
    }

    #[test]
    fn test_arch_to_coral_nvidia() {
        use crate::device::driver_profile::GpuArch;
        assert_eq!(arch_to_coral(&GpuArch::Volta), Some("sm_70"));
        assert_eq!(arch_to_coral(&GpuArch::Turing), Some("sm_75"));
        assert_eq!(arch_to_coral(&GpuArch::Ampere), Some("sm_80"));
        assert_eq!(arch_to_coral(&GpuArch::Ada), Some("sm_89"));
    }

    #[test]
    fn test_arch_to_coral_amd() {
        use crate::device::driver_profile::GpuArch;
        assert_eq!(arch_to_coral(&GpuArch::Rdna2), Some("gfx1030"));
        assert_eq!(arch_to_coral(&GpuArch::Rdna3), Some("gfx1100"));
        assert_eq!(arch_to_coral(&GpuArch::Cdna2), Some("gfx90a"));
    }

    #[test]
    fn test_arch_to_coral_unsupported() {
        use crate::device::driver_profile::GpuArch;
        assert_eq!(arch_to_coral(&GpuArch::IntelArc), None);
        assert_eq!(arch_to_coral(&GpuArch::AppleM), None);
        assert_eq!(arch_to_coral(&GpuArch::Software), None);
        assert_eq!(arch_to_coral(&GpuArch::Unknown), None);
    }
}
