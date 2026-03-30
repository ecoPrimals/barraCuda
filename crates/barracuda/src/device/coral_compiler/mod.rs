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
//! ## Module structure
//!
//! - [`types`] — wire types and arch mapping
//! - [`discovery`] — capability-based runtime discovery
//! - [`cache`] — native binary cache
//! - `jsonrpc` (internal) — low-level JSON-RPC 2.0 transport
//!
//! Fully optional: if coralReef is unavailable, all methods return `None`
//! and the standard wgpu/SovereignCompiler path is used.

pub mod cache;
pub mod discovery;
mod jsonrpc;
pub mod types;

pub use cache::{
    cache_native_binary, cached_native_binary, cached_native_binary_any_arch, shader_hash,
};
pub use discovery::discover_shader_compiler;
pub use types::{
    AdapterDescriptor, CoralBinary, CoralCapabilitiesResponse, CoralF64Capabilities, HealthResponse,
};

/// Synchronous check: can we discover a coralReef shader-compiler endpoint?
///
/// Tries env-based, capability-file-based, and (if configured) port-based
/// discovery without blocking on a full compile round-trip.  Used by
/// `SovereignDevice::new()` at startup to decide whether the sovereign
/// compile path is viable.
#[must_use]
pub fn is_coral_available() -> bool {
    if let Ok(h) = tokio::runtime::Handle::try_current() {
        let can_block = std::panic::catch_unwind(|| {
            tokio::task::block_in_place(|| {});
        })
        .is_ok();
        if can_block {
            return tokio::task::block_in_place(|| {
                h.block_on(async { discover_shader_compiler().await.is_some() })
            });
        }
    }

    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map(|r| r.block_on(async { discover_shader_compiler().await.is_some() }))
        .unwrap_or(false)
}

use jsonrpc::{jsonrpc_call, wgsl_to_spirv};
use std::sync::Arc;
use tokio::sync::RwLock;
use types::{CompileRequest, CompileResponse, CompileWgslRequest, PrecisionAdvice};

/// Connection state for the coralReef IPC client.
#[derive(Debug)]
enum ConnectionState {
    /// Haven't tried connecting yet.
    Uninit,
    /// Connected to a JSON-RPC endpoint.
    Connected { addr: Arc<str> },
    /// Discovery failed or connection refused — don't retry until reset.
    Unavailable,
}

/// IPC client for the coralReef shader compiler primal.
///
/// Lazily discovers coralReef on first use. Thread-safe via interior `RwLock`
/// — concurrent compile requests share the read lock while discovery and
/// reset take the write lock.
#[derive(Debug)]
pub struct CoralCompiler {
    state: Arc<RwLock<ConnectionState>>,
}

impl CoralCompiler {
    /// Create a new compiler client (no connection attempt until first use).
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Arc::new(RwLock::new(ConnectionState::Uninit)),
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
                binary: resp.into_bytes(),
                arch: arch.to_owned(),
            }),
            Err(e) => {
                tracing::debug!("shader compile failed: {e}");
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

        let fp64_strategy = if fp64_software {
            Some("native".to_owned())
        } else {
            Some("f32_only".to_owned())
        };
        let request = CompileWgslRequest {
            wgsl_source: wgsl.to_owned(),
            arch: arch.to_owned(),
            opt_level: 2,
            fp64_software,
            fp64_strategy,
            adapter: None,
            precision_advice: None,
        };

        match jsonrpc_call::<CompileWgslRequest, CompileResponse>(
            &addr,
            "shader.compile.wgsl",
            &request,
        )
        .await
        {
            Ok(resp) => Some(CoralBinary {
                binary: resp.into_bytes(),
                arch: arch.to_owned(),
            }),
            Err(e) => {
                tracing::debug!("shader compile_wgsl direct failed: {e}, trying SPIR-V path");
                self.compile_wgsl(wgsl, arch, fp64_software).await
            }
        }
    }

    /// Compile WGSL with precision routing advice for informed lowering.
    ///
    /// Extends [`compile_wgsl_direct`] with `PrecisionAdvice` metadata that
    /// tells coralReef whether f64 transcendental lowering is needed and
    /// which physics domain motivated the compilation.
    pub async fn compile_wgsl_with_advice(
        &self,
        wgsl: &str,
        arch: &str,
        fp64_software: bool,
        advice: PrecisionAdvice,
    ) -> Option<CoralBinary> {
        let addr = self.ensure_connected().await?;

        let fp64_strategy = if fp64_software {
            Some("native".to_owned())
        } else {
            Some("f32_only".to_owned())
        };
        let request = CompileWgslRequest {
            wgsl_source: wgsl.to_owned(),
            arch: arch.to_owned(),
            opt_level: 2,
            fp64_software,
            fp64_strategy,
            adapter: None,
            precision_advice: Some(advice),
        };

        match jsonrpc_call::<CompileWgslRequest, CompileResponse>(
            &addr,
            "shader.compile.wgsl",
            &request,
        )
        .await
        {
            Ok(resp) => Some(CoralBinary {
                binary: resp.into_bytes(),
                arch: arch.to_owned(),
            }),
            Err(e) => {
                tracing::debug!("shader compile_wgsl with advice failed: {e}, trying basic path");
                self.compile_wgsl_direct(wgsl, arch, fp64_software).await
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

    /// Query structured capabilities including f64 transcendental polyfill info.
    ///
    /// Returns the full `CoralCapabilitiesResponse` with per-op f64 lowering
    /// availability. Falls back to `None` if coralReef is unavailable or the
    /// endpoint returns the legacy flat arch list format.
    pub async fn capabilities_structured(&self) -> Option<CoralCapabilitiesResponse> {
        let addr = self.ensure_connected().await?;
        jsonrpc_call::<(), CoralCapabilitiesResponse>(&addr, "shader.compile.capabilities", &())
            .await
            .ok()
    }

    /// Query whether coralReef can provide f64 transcendental lowering.
    ///
    /// Convenience wrapper over [`capabilities_structured`] that returns `true`
    /// when coralReef reports full composite lowering for all f64 transcendentals.
    pub async fn has_f64_lowering(&self) -> bool {
        self.capabilities_structured()
            .await
            .is_some_and(|c| c.f64_transcendental_capabilities.has_full_lowering())
    }

    /// Check if coralReef is reachable and healthy via `shader.compile.status`.
    pub async fn health(&self) -> Option<HealthResponse> {
        let addr = self.ensure_connected().await?;
        jsonrpc_call::<(), HealthResponse>(&addr, "shader.compile.status", &())
            .await
            .ok()
    }

    /// Ensure we have a connection, discovering coralReef if needed.
    ///
    /// Fast path (read lock): returns the cached address if already connected.
    /// Slow path (write lock): performs discovery on first use.
    async fn ensure_connected(&self) -> Option<Arc<str>> {
        {
            let state = self.state.read().await;
            match &*state {
                ConnectionState::Connected { addr } => return Some(Arc::clone(addr)),
                ConnectionState::Unavailable => return None,
                ConnectionState::Uninit => {}
            }
        }

        let mut state = self.state.write().await;
        // Re-check after acquiring write lock (another task may have connected).
        match &*state {
            ConnectionState::Connected { addr } => return Some(Arc::clone(addr)),
            ConnectionState::Unavailable => return None,
            ConnectionState::Uninit => {}
        }

        if let Some(addr) = discover_shader_compiler().await {
            tracing::info!(addr = %addr, "discovered shader compiler service");
            let addr: Arc<str> = Arc::from(addr);
            *state = ConnectionState::Connected {
                addr: Arc::clone(&addr),
            };
            Some(addr)
        } else {
            tracing::debug!("shader compiler not available — using standard compilation path");
            *state = ConnectionState::Unavailable;
            None
        }
    }

    /// Reset connection state, forcing re-discovery on next use.
    pub async fn reset(&self) {
        let mut state = self.state.write().await;
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

/// Fire-and-forget coralReef compilation using adapter info.
///
/// Spawns a background task that queries coralReef for the compilation target
/// matching this adapter, then compiles via IPC and caches the result.
/// barraCuda does not embed per-generation ISA knowledge — coralReef determines
/// the ISA target from the adapter descriptor.
pub fn spawn_coral_compile_for_adapter(
    optimized_wgsl: &str,
    adapter_info: &wgpu::AdapterInfo,
    fp64_software: bool,
) {
    let source = optimized_wgsl.to_owned();
    let hash = shader_hash(&source);
    let adapter = types::AdapterDescriptor::from_adapter_info(adapter_info);
    let adapter_key = adapter.cache_key();

    if cached_native_binary(&hash, &adapter_key).is_some() {
        return;
    }

    let Ok(handle) = tokio::runtime::Handle::try_current() else {
        return;
    };

    handle.spawn(async move {
        let Some(archs) = GLOBAL_CORAL.supported_archs().await else {
            return;
        };

        let Some(target) = best_target_for_adapter(&archs, &adapter) else {
            return;
        };

        if let Some(binary) = GLOBAL_CORAL
            .compile_wgsl_direct(&source, &target, fp64_software)
            .await
        {
            tracing::debug!(
                target = %target,
                size = binary.binary.len(),
                "cached native binary ({} bytes) for adapter {}",
                binary.binary.len(),
                adapter.device_name,
            );
            cache_native_binary(&hash, &adapter_key, binary);
        }
    });
}

/// Select the best compilation target from coralReef's supported architectures
/// for the given adapter. Uses ISA family prefix matching (sm_ for NVIDIA,
/// gfx for AMD) — the per-generation version knowledge lives in coralReef.
fn best_target_for_adapter(
    supported_archs: &[String],
    adapter: &types::AdapterDescriptor,
) -> Option<String> {
    use crate::device::vendor::{VENDOR_AMD, VENDOR_NVIDIA};

    let prefix = match adapter.vendor_id {
        VENDOR_NVIDIA => "sm_",
        VENDOR_AMD => "gfx",
        _ => return None,
    };

    // From coralReef's supported list, find all matching the vendor's ISA family.
    // Pick the lowest (most compatible) version — GPU hardware is forward-compatible
    // with binaries compiled for earlier ISA revisions.
    supported_archs
        .iter()
        .filter(|a| a.starts_with(prefix))
        .min()
        .cloned()
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
            binary: bytes::Bytes::from_static(&[0xDE, 0xAD]),
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
    async fn test_discovery_graceful_without_shader_compiler() {
        let addr = discover_shader_compiler().await;
        if let Some(ref a) = addr {
            assert!(!a.is_empty(), "discovered address must be non-empty");
        }
    }

    #[tokio::test]
    async fn test_compile_wgsl_graceful_without_coralreef() {
        let cc = CoralCompiler::new();
        let result = cc
            .compile_wgsl("@compute @workgroup_size(64) fn main() {}", "sm_70", true)
            .await;
        if let Some(ref bin) = result {
            assert!(!bin.binary.is_empty(), "compiled binary must be non-empty");
            assert_eq!(bin.arch, "sm_70");
        }
    }

    #[tokio::test]
    async fn test_compile_wgsl_direct_graceful_without_coralreef() {
        let cc = CoralCompiler::new();
        let result = cc
            .compile_wgsl_direct("@compute @workgroup_size(64) fn main() {}", "sm_70", false)
            .await;
        if let Some(ref bin) = result {
            assert!(!bin.binary.is_empty(), "compiled binary must be non-empty");
            assert_eq!(bin.arch, "sm_70");
        }
    }

    #[tokio::test]
    async fn test_supported_archs_graceful_without_coralreef() {
        let cc = CoralCompiler::new();
        let archs = cc.supported_archs().await;
        if let Some(ref list) = archs {
            assert!(
                !list.is_empty(),
                "arch list must be non-empty when available"
            );
            for arch in list {
                assert!(!arch.is_empty(), "each arch string must be non-empty");
            }
        }
    }

    #[tokio::test]
    async fn test_reset_allows_rediscovery() {
        let cc = CoralCompiler::new();
        let health = cc.health().await;
        if let Some(ref h) = health {
            assert!(
                h.name == "coralReef" || h.name == "coralreef-core",
                "unexpected primal name: {}",
                h.name
            );
            assert!(!h.version.is_empty());
        }
        cc.reset().await;
        let state = cc.state.read().await;
        assert!(matches!(&*state, ConnectionState::Uninit));
    }

    #[tokio::test]
    async fn test_capabilities_graceful_without_coralreef() {
        let cc = CoralCompiler::new();
        let caps = cc.capabilities().await;
        if let Some(ref list) = caps {
            assert!(!list.is_empty(), "capabilities list must be non-empty");
        }
    }

    #[tokio::test]
    async fn test_connection_state_transitions() {
        let cc = CoralCompiler::new();
        {
            let state = cc.state.read().await;
            assert!(matches!(&*state, ConnectionState::Uninit));
        }
        let _ = cc.health().await;
        {
            let state = cc.state.read().await;
            assert!(
                matches!(&*state, ConnectionState::Connected { .. })
                    || matches!(&*state, ConnectionState::Unavailable)
            );
        }
    }

    #[test]
    fn test_coral_f64_capabilities_full() {
        let caps = types::CoralF64Capabilities {
            sin: true,
            cos: true,
            sqrt: true,
            exp2: true,
            log2: true,
            rcp: true,
            exp: true,
            log: true,
            composite_lowering: true,
        };
        assert!(caps.has_full_lowering());
    }

    #[test]
    fn test_coral_f64_capabilities_partial() {
        let mut caps = types::CoralF64Capabilities {
            sin: true,
            cos: true,
            sqrt: true,
            exp2: true,
            log2: true,
            rcp: true,
            exp: true,
            log: true,
            composite_lowering: false,
        };
        assert!(
            !caps.has_full_lowering(),
            "composite_lowering=false should fail"
        );
        caps.composite_lowering = true;
        caps.sin = false;
        assert!(!caps.has_full_lowering(), "sin=false should fail");
    }

    #[test]
    fn test_coral_f64_capabilities_default_empty() {
        let caps = types::CoralF64Capabilities::default();
        assert!(!caps.has_full_lowering());
    }

    #[test]
    fn test_coral_capabilities_response_json_roundtrip() {
        let resp = types::CoralCapabilitiesResponse {
            supported_archs: vec!["sm_70".to_owned(), "gfx1030".to_owned()],
            f64_transcendental_capabilities: types::CoralF64Capabilities {
                sin: true,
                cos: true,
                sqrt: true,
                exp2: true,
                log2: true,
                rcp: true,
                exp: true,
                log: true,
                composite_lowering: true,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        let back: types::CoralCapabilitiesResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(back.supported_archs.len(), 2);
        assert!(back.f64_transcendental_capabilities.has_full_lowering());
    }

    #[test]
    fn test_precision_advice_json_roundtrip() {
        let advice = types::PrecisionAdvice {
            tier: "F64".to_owned(),
            needs_transcendental_lowering: true,
            df64_naga_poisoned: true,
            domain: Some("LatticeQcd".to_owned()),
        };
        let json = serde_json::to_string(&advice).unwrap();
        let back: types::PrecisionAdvice = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tier, "F64");
        assert!(back.needs_transcendental_lowering);
        assert!(back.df64_naga_poisoned);
        assert_eq!(back.domain.as_deref(), Some("LatticeQcd"));
    }

    #[tokio::test]
    async fn test_capabilities_structured_graceful_without_coralreef() {
        let cc = CoralCompiler::new();
        let caps = cc.capabilities_structured().await;
        if let Some(ref c) = caps {
            assert!(!c.supported_archs.is_empty());
        }
    }

    #[tokio::test]
    async fn test_has_f64_lowering_graceful_without_coralreef() {
        let cc = CoralCompiler::new();
        let _ = cc.has_f64_lowering().await;
    }

    #[test]
    fn test_adapter_descriptor_cache_key() {
        let desc = types::AdapterDescriptor {
            vendor_id: 0x10DE,
            device_name: "NVIDIA GeForce RTX 3090".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
        };
        let key = desc.cache_key();
        assert!(key.starts_with("adapter:10de:"));
        assert!(key.contains("RTX 3090"));
    }

    #[test]
    fn test_best_target_for_adapter_nvidia() {
        let archs = vec!["sm_70".to_owned(), "sm_80".to_owned(), "gfx1030".to_owned()];
        let nvidia_adapter = types::AdapterDescriptor {
            vendor_id: 0x10DE,
            device_name: "NVIDIA GPU".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
        };
        let target = best_target_for_adapter(&archs, &nvidia_adapter);
        assert!(target.is_some());
        assert!(target.unwrap().starts_with("sm_"));
    }

    #[test]
    fn test_best_target_for_adapter_amd() {
        let archs = vec![
            "sm_70".to_owned(),
            "gfx1030".to_owned(),
            "gfx1100".to_owned(),
        ];
        let amd_adapter = types::AdapterDescriptor {
            vendor_id: 0x1002,
            device_name: "AMD Radeon".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
        };
        let target = best_target_for_adapter(&archs, &amd_adapter);
        assert!(target.is_some());
        assert!(target.unwrap().starts_with("gfx"));
    }

    #[test]
    fn test_best_target_for_adapter_unsupported() {
        let archs = vec!["sm_70".to_owned()];
        let intel_adapter = types::AdapterDescriptor {
            vendor_id: 0x8086,
            device_name: "Intel Arc".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
        };
        assert!(best_target_for_adapter(&archs, &intel_adapter).is_none());
    }

    // ── cache tests ─────────────────────────────────────────────────────

    #[test]
    fn cache_insert_and_lookup() {
        let hash = cache::shader_hash("test_shader_source_1234");
        let binary = types::CoralBinary {
            binary: bytes::Bytes::from_static(&[0xCA, 0xFE]),
            arch: "sm_70".to_owned(),
        };
        cache::cache_native_binary(&hash, "sm_70", binary);
        let found = cache::cached_native_binary(&hash, "sm_70");
        assert!(found.is_some());
        assert_eq!(found.unwrap().arch, "sm_70");
    }

    #[test]
    fn cache_miss_returns_none() {
        assert!(cache::cached_native_binary("nonexistent_hash_xyz", "sm_70").is_none());
    }

    #[test]
    fn cache_any_arch_finds_first_match() {
        let hash = cache::shader_hash("any_arch_test_source_5678");
        cache::cache_native_binary(
            &hash,
            "gfx1030",
            types::CoralBinary {
                binary: bytes::Bytes::from_static(&[0xAA]),
                arch: "gfx1030".to_owned(),
            },
        );
        let found = cache::cached_native_binary_any_arch(&hash);
        assert!(found.is_some());
        assert_eq!(found.unwrap().arch, "gfx1030");
    }

    #[test]
    fn cache_any_arch_miss() {
        assert!(cache::cached_native_binary_any_arch("completely_missing_hash").is_none());
    }

    #[test]
    fn shader_hash_deterministic() {
        let h1 = cache::shader_hash("hello world");
        let h2 = cache::shader_hash("hello world");
        assert_eq!(h1, h2);
        let h3 = cache::shader_hash("hello world!");
        assert_ne!(h1, h3);
    }

    #[test]
    fn shader_hash_is_hex() {
        let h = cache::shader_hash("some shader code");
        assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(h.len(), 64);
    }

    // ── types serialization tests ───────────────────────────────────────

    #[test]
    fn adapter_descriptor_json_roundtrip() {
        let desc = types::AdapterDescriptor {
            vendor_id: 0x10DE,
            device_name: "Test GPU".to_owned(),
            device_type: "DiscreteGpu".to_owned(),
        };
        let json = serde_json::to_string(&desc).unwrap();
        let back: types::AdapterDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(back.vendor_id, 0x10DE);
        assert_eq!(back.device_name, "Test GPU");
        assert_eq!(back.device_type, "DiscreteGpu");
    }

    #[test]
    fn health_response_deserialize() {
        let json = r#"{"name":"coralReef","version":"0.3.0","status":"healthy","supported_archs":["sm_70","sm_80"]}"#;
        let resp: types::HealthResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.name, "coralReef");
        assert_eq!(resp.supported_archs.len(), 2);
    }

    #[test]
    fn precision_to_coral_strategy_all_variants() {
        use crate::shaders::precision::Precision;
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Binary),
            "binary"
        );
        assert_eq!(types::precision_to_coral_strategy(&Precision::Int2), "int2");
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Q4),
            "q4_block"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Q8),
            "q8_block"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Fp8E5M2),
            "fp8_e5m2"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Fp8E4M3),
            "fp8_e4m3"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Bf16),
            "bf16_emulated"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::F16),
            "f16_fast"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::F32),
            "f32_only"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::F64),
            "native"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Df64),
            "double_float"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Qf128),
            "quad_float"
        );
        assert_eq!(
            types::precision_to_coral_strategy(&Precision::Df128),
            "double_double_f64"
        );
    }
}
