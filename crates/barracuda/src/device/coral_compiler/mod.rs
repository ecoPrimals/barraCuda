// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader compiler IPC client — discovers a primal via the `shader.compile` capability.
//!
//! Connects to a discovered shader compiler primal's JSON-RPC 2.0 endpoint
//! (not a hardcoded primal name), providing native GPU binary compilation for
//! NVIDIA (SM70+) and AMD (RDNA2+) architectures. **coralReef** is the reference
//! implementation of this wire protocol.
//!
//! ## IPC Contract (Phase 10 + Phase 3 CPU; coralReef reference)
//!
//! | Method | Purpose |
//! |--------|---------|
//! | `shader.compile.spirv` | SPIR-V → native GPU binary |
//! | `shader.compile.wgsl` | WGSL → native GPU binary |
//! | `shader.compile.cpu` | WGSL → native CPU binary (Phase 3) |
//! | `shader.execute.cpu` | WGSL + buffers → executed results on CPU (Phase 3) |
//! | `shader.validate` | WGSL + inputs + expected → validation report (Phase 3) |
//! | `shader.compile.status` | Health / readiness |
//! | `shader.compile.capabilities` | Supported architectures + CPU/validation flags |
//!
//! ## Module structure
//!
//! - [`types`] — wire types and arch mapping
//! - [`discovery`] — capability-based runtime discovery
//! - [`cache`] — native binary cache
//! - `jsonrpc` (internal) — low-level JSON-RPC 2.0 transport
//!
//! Fully optional: if no shader compiler primal is discovered, all methods return `None`
//! and the standard wgpu/SovereignCompiler path is used.

pub mod cache;
pub mod discovery;
mod jsonrpc;
pub mod types;

pub use cache::{
    cache_native_binary, cached_native_binary, cached_native_binary_any_arch, shader_hash,
};
pub use discovery::{
    DEFAULT_ECOPRIMALS_DISCOVERY_DIR, discover_cpu_shader_compiler, discover_shader_compiler,
    discover_shader_validator,
};
pub use types::{
    AdapterDescriptor, BufferBinding, CompileCpuRequest, CompileCpuResponse, CoralBinary,
    CoralCapabilitiesResponse, CoralF64Capabilities, ExecuteCpuRequest, ExecuteCpuResponse,
    ExpectedBinding, HealthResponse, ValidateRequest, ValidateResponse, ValidationMismatch,
    ValidationTolerance,
};

/// Synchronous check: can we discover a shader-compiler endpoint (`shader.compile`)?
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

/// Connection state for the shader compiler IPC client.
#[derive(Debug)]
enum ConnectionState {
    /// Haven't tried connecting yet.
    Uninit,
    /// Connected to a JSON-RPC endpoint.
    Connected { addr: Arc<str> },
    /// Discovery failed or connection refused — don't retry until reset.
    Unavailable,
}

/// IPC client for a shader compiler primal (discovered by `shader.compile` capability).
///
/// Lazily discovers the endpoint on first use. Thread-safe via interior `RwLock`
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

    /// Attempt to compile WGSL to a native GPU binary via the discovered compiler.
    ///
    /// Returns `None` if no compiler is available or compilation fails.
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
    /// Returns `None` if no compiler is available or compilation fails.
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

    /// Compile WGSL directly via Phase 10 `shader.compile.wgsl`.
    ///
    /// Unlike [`compile_wgsl`], this sends raw WGSL to the remote compiler for
    /// server-side compilation, avoiding the local naga WGSL → SPIR-V step.
    /// The compiler handles the full pipeline: WGSL → IR → native binary.
    ///
    /// Falls back to the SPIR-V path if the direct endpoint is unavailable
    /// (older implementations).
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
    /// tells the remote compiler whether f64 transcendental lowering is needed and
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

    /// Query supported GPU architectures from the discovered compiler.
    ///
    /// Prefers the Phase 10 `shader.compile.capabilities` endpoint, falling
    /// back to `shader.compile.status` for older implementations that
    /// embed arch info in the health response.
    pub async fn supported_archs(&self) -> Option<Vec<String>> {
        if let Some(archs) = self.capabilities().await {
            return Some(archs);
        }
        self.health().await.map(|h| h.supported_archs)
    }

    /// Query supported architectures via `shader.compile.capabilities`.
    ///
    /// Returns `None` if no compiler is available or the endpoint is not
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
    /// availability. Falls back to `None` if no compiler is available or the
    /// endpoint returns the legacy flat arch list format.
    pub async fn capabilities_structured(&self) -> Option<CoralCapabilitiesResponse> {
        let addr = self.ensure_connected().await?;
        jsonrpc_call::<(), CoralCapabilitiesResponse>(&addr, "shader.compile.capabilities", &())
            .await
            .ok()
    }

    /// Query whether the discovered compiler can provide f64 transcendental lowering.
    ///
    /// Convenience wrapper over [`capabilities_structured`] that returns `true`
    /// when the endpoint reports full composite lowering for all f64 transcendentals.
    pub async fn has_f64_lowering(&self) -> bool {
        self.capabilities_structured()
            .await
            .is_some_and(|c| c.f64_transcendental_capabilities.has_full_lowering())
    }

    /// Check if the discovered compiler is reachable and healthy via `shader.compile.status`.
    pub async fn health(&self) -> Option<HealthResponse> {
        let addr = self.ensure_connected().await?;
        jsonrpc_call::<(), HealthResponse>(&addr, "shader.compile.status", &())
            .await
            .ok()
    }

    // ========================================================================
    // Phase 3: CPU compilation, execution, and validation
    // ========================================================================

    /// Whether the discovered compiler supports CPU shader execution (`shader.execute.cpu`).
    ///
    /// Queries the structured capabilities endpoint and checks for the
    /// `supports_cpu_execution` flag. Returns `false` if no compiler is
    /// available or it doesn't advertise CPU capabilities.
    pub async fn supports_cpu_execution(&self) -> bool {
        self.capabilities_structured()
            .await
            .is_some_and(|c| c.supports_cpu_execution)
    }

    /// Whether the discovered compiler supports shader validation (`shader.validate`).
    pub async fn supports_validation(&self) -> bool {
        self.capabilities_structured()
            .await
            .is_some_and(|c| c.supports_validation)
    }

    /// Compile WGSL to a native CPU binary via `shader.compile.cpu`.
    ///
    /// Returns `None` if no compiler is available or CPU compilation is unsupported.
    /// The binary is a native executable for the specified
    /// architecture (`x86_64`, `aarch64`, or `auto`-detect).
    pub async fn compile_cpu(
        &self,
        wgsl: &str,
        entry_point: &str,
        arch: &str,
    ) -> Option<types::CompileCpuResponse> {
        let addr = self.ensure_connected().await?;
        let request = types::CompileCpuRequest {
            wgsl_source: wgsl.to_owned(),
            arch: arch.to_owned(),
            opt_level: 2,
            entry_point: entry_point.to_owned(),
        };
        match jsonrpc_call::<types::CompileCpuRequest, types::CompileCpuResponse>(
            &addr,
            "shader.compile.cpu",
            &request,
        )
        .await
        {
            Ok(resp) => Some(resp),
            Err(e) => {
                tracing::debug!("shader.compile.cpu failed: {e}");
                None
            }
        }
    }

    /// Execute a WGSL shader on CPU via `shader.execute.cpu`.
    ///
    /// Compiles and runs the shader in one IPC call. The remote compiler handles
    /// compilation, dispatch simulation, and buffer read-back.
    ///
    /// Returns `None` if no compiler is available or execution fails.
    pub async fn execute_cpu(
        &self,
        request: &types::ExecuteCpuRequest,
    ) -> Option<types::ExecuteCpuResponse> {
        let addr = self.ensure_connected().await?;
        match jsonrpc_call::<types::ExecuteCpuRequest, types::ExecuteCpuResponse>(
            &addr,
            "shader.execute.cpu",
            request,
        )
        .await
        {
            Ok(resp) => Some(resp),
            Err(e) => {
                tracing::debug!("shader.execute.cpu failed: {e}");
                None
            }
        }
    }

    /// Validate a WGSL shader's output against expected values via `shader.validate`.
    ///
    /// The compiler executes the shader on CPU and compares each output element
    /// against the expected values with the specified tolerances. Returns a
    /// validation report with pass/fail status and per-element mismatches.
    ///
    /// Returns `None` if no compiler is available or the validation endpoint
    /// is not supported.
    pub async fn validate_shader(
        &self,
        request: &types::ValidateRequest,
    ) -> Option<types::ValidateResponse> {
        let addr = self.ensure_connected().await?;
        match jsonrpc_call::<types::ValidateRequest, types::ValidateResponse>(
            &addr,
            "shader.validate",
            request,
        )
        .await
        {
            Ok(resp) => Some(resp),
            Err(e) => {
                tracing::debug!("shader.validate failed: {e}");
                None
            }
        }
    }

    /// Ensure we have a connection, running capability-based discovery if needed.
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

/// Global shader compiler IPC client (lazy singleton like `GLOBAL_CACHE`).
pub static GLOBAL_CORAL: std::sync::LazyLock<CoralCompiler> =
    std::sync::LazyLock::new(CoralCompiler::new);

/// Get a reference to the global shader compiler client.
///
/// Lazily initialized on first access. Thread-safe.
#[must_use]
pub fn global_coral() -> &'static CoralCompiler {
    &GLOBAL_CORAL
}

/// Lightweight health probe for a discovered shader compiler.
///
/// Uses the global singleton — does not create a new `CoralCompiler` instance.
/// Returns `true` if the endpoint is reachable and reports healthy status.
pub async fn probe_health() -> bool {
    GLOBAL_CORAL.health().await.is_some()
}

/// Fire-and-forget compilation using adapter info (discovered `shader.compile` endpoint).
///
/// Spawns a background task that queries the compiler for the compilation target
/// matching this adapter, then compiles via IPC and caches the result.
/// barraCuda does not embed per-generation ISA knowledge — the remote compiler determines
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

/// Select the best compilation target from the compiler's supported architectures
/// for the given adapter. Uses ISA family prefix matching (sm_ for NVIDIA,
/// gfx for AMD) — the per-generation version knowledge lives in the remote compiler.
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

    // From the compiler's supported list, find all matching the vendor's ISA family.
    // Pick the lowest (most compatible) version — GPU hardware is forward-compatible
    // with binaries compiled for earlier ISA revisions.
    supported_archs
        .iter()
        .filter(|a| a.starts_with(prefix))
        .min()
        .cloned()
}

#[cfg(test)]
#[path = "coral_compiler_tests.rs"]
mod tests;
