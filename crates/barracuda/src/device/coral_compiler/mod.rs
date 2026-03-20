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

pub use cache::{cache_native_binary, cached_native_binary, shader_hash};
pub use discovery::discover_shader_compiler;
pub use types::{CoralBinary, HealthResponse, arch_to_coral};

/// Synchronous check: can we discover a coralReef shader-compiler endpoint?
///
/// Tries env-based, capability-file-based, and (if configured) port-based
/// discovery without blocking on a full compile round-trip.  Used by
/// `CoralReefDevice::new()` at startup to decide whether the sovereign
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
use types::{CompileRequest, CompileResponse, CompileWgslRequest};

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
                "cached native binary ({} bytes)",
                binary.binary.len(),
            );
            cache_native_binary(&hash, &arch_owned, binary);
        }
    });
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
    fn test_arch_to_coral_nvidia() {
        use crate::device::driver_profile::GpuArch;
        assert_eq!(arch_to_coral(&GpuArch::Volta), Some("sm_70"));
        assert_eq!(arch_to_coral(&GpuArch::Turing), Some("sm_75"));
        assert_eq!(arch_to_coral(&GpuArch::Ampere), Some("sm_80"));
        assert_eq!(arch_to_coral(&GpuArch::Ada), Some("sm_89"));
        assert_eq!(arch_to_coral(&GpuArch::Blackwell), Some("sm_100"));
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
