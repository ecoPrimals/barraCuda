// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign compute backend via capability-based IPC discovery.
//!
//! `SovereignDevice` is the GPU backend for the sovereign pipeline.
//! It uses JSON-RPC IPC to communicate with ecosystem primals at runtime:
//!
//! - A primal advertising `shader.compile` compiles WGSL → native GPU binaries
//! - A primal advertising `compute.dispatch` dispatches binaries to hardware
//!
//! No compile-time coupling to any other primal crate. barraCuda is one math
//! library for any deployment — the GPU backend is determined entirely at
//! runtime by capability-based discovery.
//!
//! # Architecture (IPC-first)
//!
//! ```text
//! barraCuda (WGSL math)
//!   → [JSON-RPC] shader.compile primal (WGSL → native SASS/GFX binary)
//!     → [JSON-RPC] compute.dispatch primal (dispatch binary → VFIO/DRM → GPU)
//!       → GPU hardware (any PCIe GPU — NVIDIA, AMD, Intel)
//! ```
//!
//! barraCuda never touches VFIO, DRM, or GPU hardware directly.
//! The dispatch primal owns all hardware lifecycle (VFIO bind/unbind, DMA, thermal).
//! The compile primal owns all compilation (naga → native binary, f64 lowering).
//!
//! # Backend maturity
//!
//! | Path | Status | Notes |
//! |------|--------|-------|
//! | IPC compile (`shader.compile`) | Done | via `coral_compiler/` |
//! | IPC dispatch (`compute.dispatch`) | Wired | `compute.dispatch.submit` (S152); readback pending |
//! | DRM (nouveau/amdgpu) | E2E verified | Titan V + RTX 3090 proven |
//! | VFIO/GPFIFO | Fix applied | USERD_TARGET + INST_TARGET fix (Iter 44); hw revalidation pending |
//!
//! # Activation
//!
//! ```toml
//! [features]
//! sovereign-dispatch = ["gpu"]
//! ```

use super::backend::{BufferBinding, DispatchDescriptor, GpuBackend};
use super::coral_compiler::DEFAULT_ECOPRIMALS_DISCOVERY_DIR;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

#[cfg(feature = "sovereign-dispatch")]
use std::collections::HashMap;

#[cfg(feature = "sovereign-dispatch")]
const CONSERVATIVE_GPR_COUNT: u32 = 128;

#[cfg(feature = "sovereign-dispatch")]
const DEFAULT_WORKGROUP: [u32; 3] = [64, 1, 1];

#[cfg(feature = "sovereign-dispatch")]
static GPR_COUNT: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
    std::env::var("BARRACUDA_GPR_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(CONSERVATIVE_GPR_COUNT)
});

#[cfg(feature = "sovereign-dispatch")]
static RESOLVED_DEFAULT_WORKGROUP: std::sync::LazyLock<[u32; 3]> = std::sync::LazyLock::new(|| {
    let x = std::env::var("BARRACUDA_DEFAULT_WORKGROUP_X")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_WORKGROUP[0]);
    [x, 1, 1]
});

/// Capability string for compute dispatch discovery.
const DISPATCH_CAPABILITY: &str = "compute.dispatch";

/// Environment variable for explicit compute dispatch endpoint override.
const DISPATCH_ADDR_ENV: &str = "BARRACUDA_DISPATCH_ADDR";

/// Canonical discovery subdirectory name.
const DISCOVERY_SUBDIR: &str = "discovery";

/// Metadata about a compiled shader needed for GPU dispatch.
///
/// Sent to the dispatch primal so it can configure QMD (NVIDIA) or PM4 (AMD)
/// descriptors with the correct register count and workgroup dimensions.
#[cfg(feature = "sovereign-dispatch")]
#[derive(Debug, Clone, Copy)]
struct ShaderDispatchInfo {
    gpr_count: u32,
    workgroup: [u32; 3],
}

/// Serialisable buffer binding descriptor for IPC compute dispatch.
///
/// Carries the buffer identity and access mode across the JSON-RPC boundary.
/// The dispatch primal uses `buffer_id` to resolve staged GPU memory, `size`
/// for validation, and `read_only` to set appropriate access flags.
#[cfg(feature = "sovereign-dispatch")]
#[derive(Debug, Clone)]
struct IpcBufferBinding {
    index: u32,
    buffer_id: u64,
    size: u64,
    read_only: bool,
}

/// Buffer handle for the sovereign compute path.
///
/// Wraps a dispatch-side buffer identifier. The `id` is assigned locally
/// during staging; the dispatch primal maps it to actual GPU memory.
#[derive(Debug, Clone, Copy)]
pub struct SovereignBuffer {
    #[cfg(feature = "sovereign-dispatch")]
    id: u64,
    size: u64,
}

/// Sovereign GPU compute device via capability-based IPC discovery.
///
/// Compilation flows through the `coral_compiler/` JSON-RPC client to
/// whichever primal advertises `shader.compile`. Dispatch flows through
/// `compute.dispatch.submit` to the primal advertising `compute.dispatch`.
/// Both are runtime IPC — no compile-time coupling to any primal crate.
pub struct SovereignDevice {
    name: Arc<str>,
    #[cfg(feature = "sovereign-dispatch")]
    compiler_available: bool,
    #[cfg(feature = "sovereign-dispatch")]
    dispatch_addr: Option<String>,
    /// Compiled binary cache. `std::sync::Mutex` is correct: never held across `.await`.
    #[cfg(feature = "sovereign-dispatch")]
    binary_cache: std::sync::Mutex<HashMap<u64, CachedBinary>>,
    /// Staging buffers for IPC dispatch. `std::sync::Mutex` is correct: never held across `.await`.
    #[cfg(feature = "sovereign-dispatch")]
    staged_buffers: std::sync::Mutex<HashMap<u64, bytes::BytesMut>>,
}

/// A compiled native GPU binary cached from IPC compilation.
///
/// `gpr_count` and `workgroup` are tracked for dispatch metadata.
/// They will be sent as part of the `compute.dispatch.submit` payload once
/// the dispatch primal accepts kernel metadata alongside the binary.
#[cfg(feature = "sovereign-dispatch")]
#[derive(Clone)]
struct CachedBinary {
    binary: bytes::Bytes,
    gpr_count: u32,
    workgroup: [u32; 3],
}

impl std::fmt::Debug for SovereignDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SovereignDevice")
            .field("name", &self.name)
            .finish()
    }
}

/// Detect whether a compute dispatch endpoint is available at runtime.
///
/// Discovery chain (first match wins):
/// 1. `BARRACUDA_DISPATCH_ADDR` env var
/// 2. Capability scan of `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
///    `"compute.dispatch"` in the `provides` or `capabilities` array
#[cfg(feature = "sovereign-dispatch")]
fn detect_dispatch_addr() -> Option<String> {
    if let Ok(addr) = std::env::var(DISPATCH_ADDR_ENV) {
        let addr = addr.trim().to_owned();
        if !addr.is_empty() {
            return Some(addr);
        }
    }

    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let eco_dir = std::env::var("ECOPRIMALS_DISCOVERY_DIR")
        .unwrap_or_else(|_| DEFAULT_ECOPRIMALS_DISCOVERY_DIR.to_owned());
    let base_dir = std::path::PathBuf::from(runtime_dir).join(eco_dir);

    for dir in [&base_dir, &base_dir.join(DISCOVERY_SUBDIR)] {
        if let Some(addr) = scan_dispatch_capability(dir) {
            return Some(addr);
        }
    }
    None
}

/// Scan a directory for any primal advertising `compute.dispatch`.
#[cfg(feature = "sovereign-dispatch")]
fn scan_dispatch_capability(dir: &std::path::Path) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            if let Some(addr) = read_dispatch_transport(&path) {
                return Some(addr);
            }
        }
    }
    None
}

/// Read a primal manifest and extract transport if it provides dispatch.
#[cfg(feature = "sovereign-dispatch")]
fn read_dispatch_transport(path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;

    let has_dispatch = info
        .get("provides")
        .or_else(|| info.get("capabilities"))
        .and_then(|v| v.as_array())
        .is_some_and(|caps| {
            caps.iter().any(|c| {
                c.as_str()
                    .is_some_and(|s| s.starts_with(DISPATCH_CAPABILITY))
            })
        });

    if !has_dispatch {
        return None;
    }

    let jsonrpc = info.get("transports")?.get("jsonrpc")?;
    if let Some(s) = jsonrpc.as_str() {
        return Some(s.to_owned());
    }
    jsonrpc
        .get("tcp")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

impl SovereignDevice {
    /// Create a `SovereignDevice` that compiles via `shader.compile` IPC and
    /// dispatches via `compute.dispatch` IPC.
    ///
    /// Both primals are discovered at runtime via capability-based discovery.
    /// If neither is available, the device can still serve as a compile-only
    /// backend (using the shader compiler cache for pre-compiled binaries).
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn new() -> Self {
        let compiler_available = crate::device::coral_compiler::is_coral_available();
        let dispatch_addr = detect_dispatch_addr();
        if let Some(ref addr) = dispatch_addr {
            tracing::info!(addr = %addr, "discovered compute.dispatch endpoint");
        }
        Self {
            name: Arc::from("sovereign-ipc (shader.compile + compute.dispatch)"),
            compiler_available,
            dispatch_addr,
            binary_cache: std::sync::Mutex::new(HashMap::new()),
            staged_buffers: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create a `SovereignDevice` (feature not enabled).
    ///
    /// # Errors
    ///
    /// Always returns `Err` when the `sovereign-dispatch` feature is disabled.
    #[cfg(not(feature = "sovereign-dispatch"))]
    pub fn new_disabled() -> Result<Self> {
        Err(BarracudaError::Device(
            "SovereignDevice: enable the 'sovereign-dispatch' feature — \
             cargo build --features sovereign-dispatch"
                .into(),
        ))
    }

    /// Attempt to create a `SovereignDevice` with auto-detected hardware.
    ///
    /// Returns `Ok(device)` when the `sovereign-dispatch` feature is enabled
    /// and IPC discovery succeeds. Callers check `has_dispatch()` to
    /// determine whether the device can actually dispatch compute.
    ///
    /// # Errors
    ///
    /// Returns `Err` when the `sovereign-dispatch` feature is disabled.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn with_auto_device() -> Result<Self> {
        Ok(Self::new())
    }

    /// Attempt to create a `SovereignDevice` (feature not enabled).
    ///
    /// # Errors
    ///
    /// Always returns `Err` when the `sovereign-dispatch` feature is disabled.
    #[cfg(not(feature = "sovereign-dispatch"))]
    pub fn with_auto_device() -> Result<Self> {
        Err(BarracudaError::Device(
            "SovereignDevice: enable the 'sovereign-dispatch' feature".into(),
        ))
    }

    /// Whether a shader compiler primal is available via IPC.
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn has_compiler(&self) -> bool {
        self.compiler_available
    }

    /// Whether compute dispatch is available for this device.
    ///
    /// Returns `true` when an endpoint advertising `compute.dispatch`
    /// was discovered at construction time.
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn has_dispatch(&self) -> bool {
        self.dispatch_addr.is_some()
    }

    /// Dispatch is never available without the feature.
    #[cfg(not(feature = "sovereign-dispatch"))]
    #[must_use]
    pub fn has_dispatch(&self) -> bool {
        false
    }

    /// Check the coral compiler cache for a pre-compiled native binary.
    ///
    /// Searches for any cached binary matching the shader hash, regardless of
    /// ISA target. The correct target was determined by the compiler at compile
    /// time; the dispatch primal handles hardware routing at dispatch time.
    fn try_coral_cache(shader_source: &str) -> Option<bytes::Bytes> {
        use crate::device::coral_compiler::{cache::cached_native_binary_any_arch, shader_hash};
        let hash = shader_hash(shader_source);
        cached_native_binary_any_arch(&hash).map(|b| b.binary)
    }

    /// Submit a dispatch request to the compute.dispatch primal via JSON-RPC.
    ///
    /// Sends the compiled native binary, workgroup dimensions, buffer binding
    /// descriptors (IDs + access mode), and hardware routing hint. The dispatch
    /// primal maps buffer IDs to GPU memory and routes to the appropriate
    /// hardware unit.
    #[cfg(feature = "sovereign-dispatch")]
    fn submit_dispatch(
        &self,
        binary: &[u8],
        workgroups: (u32, u32, u32),
        bindings: &[IpcBufferBinding],
        hardware_hint: super::backend::HardwareHint,
        shader_info: ShaderDispatchInfo,
    ) -> Result<()> {
        let Some(ref addr) = self.dispatch_addr else {
            return Err(BarracudaError::Device(
                "SovereignDevice: no compute.dispatch endpoint discovered — \
                 ensure a dispatch primal is running and advertising compute.dispatch"
                    .into(),
            ));
        };

        let binding_descriptors: Vec<serde_json::Value> = bindings
            .iter()
            .map(|b| {
                serde_json::json!({
                    "index": b.index,
                    "buffer_id": b.buffer_id,
                    "size": b.size,
                    "read_only": b.read_only,
                })
            })
            .collect();

        let hint_str = match hardware_hint {
            super::backend::HardwareHint::Compute => "compute",
            super::backend::HardwareHint::TensorCore => "tensor_core",
            super::backend::HardwareHint::RtCore => "rt_core",
            super::backend::HardwareHint::ZBuffer => "zbuffer",
            super::backend::HardwareHint::TextureUnit => "texture_unit",
            super::backend::HardwareHint::RopBlend => "rop_blend",
        };

        let request = serde_json::json!({
            "binary": binary,
            "workgroup_size": [workgroups.0, workgroups.1, workgroups.2],
            "bindings": binding_descriptors,
            "hardware_hint": hint_str,
            "gpr_count": shader_info.gpr_count,
            "workgroup": shader_info.workgroup,
        });

        let addr = addr.clone();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return Err(BarracudaError::Device(
                "SovereignDevice: no tokio runtime available for IPC dispatch".into(),
            ));
        };

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                let host_port = addr.trim_start_matches("http://");
                let body = serde_json::to_string(&serde_json::json!({
                    "jsonrpc": "2.0",
                    "method": "compute.dispatch.submit",
                    "params": [request],
                    "id": 1,
                }))
                .map_err(|e| BarracudaError::Device(format!("serialize dispatch: {e}")))?;

                let http_request = format!(
                    "POST / HTTP/1.1\r\nHost: {host_port}\r\n\
                     Content-Type: application/json\r\n\
                     Content-Length: {}\r\n\
                     Connection: close\r\n\r\n{body}",
                    body.len()
                );

                let mut stream = tokio::net::TcpStream::connect(host_port)
                    .await
                    .map_err(|e| {
                        BarracudaError::Device(format!(
                            "SovereignDevice: dispatch connect to {host_port}: {e}"
                        ))
                    })?;

                tokio::io::AsyncWriteExt::write_all(&mut stream, http_request.as_bytes())
                    .await
                    .map_err(|e| {
                        BarracudaError::Device(format!("SovereignDevice: dispatch write: {e}"))
                    })?;

                let mut response_buf = Vec::new();
                tokio::io::AsyncReadExt::read_to_end(&mut stream, &mut response_buf)
                    .await
                    .map_err(|e| {
                        BarracudaError::Device(format!("SovereignDevice: dispatch read: {e}"))
                    })?;

                let response_str = String::from_utf8_lossy(&response_buf);
                let json_start = response_str.find('{').ok_or_else(|| {
                    BarracudaError::Device(
                        "SovereignDevice: no JSON body in dispatch response".into(),
                    )
                })?;

                let rpc_response: serde_json::Value =
                    serde_json::from_str(&response_str[json_start..]).map_err(|e| {
                        BarracudaError::Device(format!("SovereignDevice: parse response: {e}"))
                    })?;

                if let Some(error) = rpc_response.get("error") {
                    let msg = error
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown error");
                    return Err(BarracudaError::Device(format!(
                        "SovereignDevice: compute dispatch error: {msg}"
                    )));
                }

                tracing::debug!(
                    addr = %host_port,
                    "sovereign dispatch submitted to compute.dispatch endpoint"
                );
                Ok(())
            })
        })
    }
}

#[cfg(feature = "sovereign-dispatch")]
impl Default for SovereignDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for SovereignDevice {
    type Buffer = SovereignBuffer;

    fn name(&self) -> &str {
        &self.name
    }

    fn has_f64_shaders(&self) -> bool {
        true
    }

    fn is_lost(&self) -> bool {
        false
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn alloc_buffer(&self, _label: &str, size: u64) -> Result<SovereignBuffer> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut staged) = self.staged_buffers.lock() {
            staged.insert(id, bytes::BytesMut::zeroed(size as usize));
        }
        Ok(SovereignBuffer { id, size })
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn alloc_buffer(&self, _label: &str, _size: u64) -> Result<SovereignBuffer> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn alloc_buffer_init(&self, label: &str, contents: &[u8]) -> Result<SovereignBuffer> {
        let buf = self.alloc_buffer(label, contents.len() as u64)?;
        self.upload(&buf, 0, contents);
        Ok(buf)
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn alloc_buffer_init(&self, _label: &str, _contents: &[u8]) -> Result<SovereignBuffer> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn alloc_uniform(&self, label: &str, contents: &[u8]) -> Result<SovereignBuffer> {
        self.alloc_buffer_init(label, contents)
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn alloc_uniform(&self, _label: &str, _contents: &[u8]) -> Result<SovereignBuffer> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn upload(&self, buffer: &SovereignBuffer, offset: u64, data: &[u8]) {
        if let Ok(mut staged) = self.staged_buffers.lock() {
            if let Some(buf) = staged.get_mut(&buffer.id) {
                let start = offset as usize;
                let end = start + data.len();
                if end <= buf.len() {
                    buf[start..end].copy_from_slice(data);
                }
            }
        }
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn upload(&self, _buffer: &SovereignBuffer, _offset: u64, _data: &[u8]) {}

    #[cfg(feature = "sovereign-dispatch")]
    fn download(&self, buffer: &SovereignBuffer, _size: u64) -> Result<bytes::Bytes> {
        let staged = self.staged_buffers.lock().map_err(|e| {
            BarracudaError::Device(format!("SovereignDevice: staged lock poisoned: {e}"))
        })?;
        // `copy_from_slice` is required here: staged buffers remain mutable
        // (`BytesMut`) for future `upload()` calls, so we cannot `freeze()`
        // in-place. The copy is bounded by buffer size and only occurs on
        // explicit download requests. The wgpu backend achieves true zero-copy
        // via GPU buffer mapping; this path is for the sovereign staging layer.
        staged
            .get(&buffer.id)
            .map(|b| bytes::Bytes::copy_from_slice(b))
            .ok_or_else(|| {
                BarracudaError::Device(format!(
                    "SovereignDevice: buffer {} not found in staging",
                    buffer.id
                ))
            })
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn download(&self, _buffer: &SovereignBuffer, _size: u64) -> Result<bytes::Bytes> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn dispatch_compute(&self, desc: DispatchDescriptor<'_, Self>) -> Result<()> {
        use std::collections::hash_map::Entry;
        use std::hash::{Hash, Hasher};

        let mut hasher = std::hash::DefaultHasher::new();
        desc.shader_source.hash(&mut hasher);
        let key = hasher.finish();

        let mut cache = self.binary_cache.lock().map_err(|e| {
            BarracudaError::Device(format!("SovereignDevice: cache lock poisoned: {e}"))
        })?;

        let cached = match cache.entry(key) {
            Entry::Occupied(e) => {
                tracing::debug!("sovereign cache hit — using pre-compiled native binary");
                e.into_mut()
            }
            Entry::Vacant(e) => {
                if let Some(binary) = Self::try_coral_cache(desc.shader_source) {
                    tracing::debug!("coral compiler cache hit — using pre-compiled native binary");
                    e.insert(CachedBinary {
                        binary,
                        gpr_count: *GPR_COUNT,
                        workgroup: *RESOLVED_DEFAULT_WORKGROUP,
                    })
                } else {
                    return Err(BarracudaError::Device(
                        "SovereignDevice: no cached binary and live IPC compilation \
                         not yet implemented — pre-compile via spawn_coral_compile()"
                            .into(),
                    ));
                }
            }
        };

        let binary = cached.binary.clone();
        let info = ShaderDispatchInfo {
            gpr_count: cached.gpr_count,
            workgroup: cached.workgroup,
        };
        drop(cache);

        let ipc_bindings: Vec<IpcBufferBinding> = desc
            .bindings
            .iter()
            .map(|b| IpcBufferBinding {
                index: b.index,
                buffer_id: b.buffer.id,
                size: b.buffer.size,
                read_only: b.read_only,
            })
            .collect();

        self.submit_dispatch(
            &binary,
            desc.workgroups,
            &ipc_bindings,
            desc.hardware_hint,
            info,
        )
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn dispatch_compute(&self, _desc: DispatchDescriptor<'_, Self>) -> Result<()> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn dispatch_binary(
        &self,
        binary: &[u8],
        bindings: Vec<BufferBinding<'_, Self>>,
        workgroups: (u32, u32, u32),
        _entry_point: &str,
    ) -> Result<()> {
        let ipc_bindings: Vec<IpcBufferBinding> = bindings
            .iter()
            .map(|b| IpcBufferBinding {
                index: b.index,
                buffer_id: b.buffer.id,
                size: b.buffer.size,
                read_only: b.read_only,
            })
            .collect();

        self.submit_dispatch(
            binary,
            workgroups,
            &ipc_bindings,
            super::backend::HardwareHint::Compute,
            ShaderDispatchInfo {
                gpr_count: *GPR_COUNT,
                workgroup: *RESOLVED_DEFAULT_WORKGROUP,
            },
        )
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn dispatch_binary(
        &self,
        _binary: &[u8],
        _bindings: Vec<BufferBinding<'_, Self>>,
        _workgroups: (u32, u32, u32),
        _entry_point: &str,
    ) -> Result<()> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn device_creation_ipc() {
        let dev = SovereignDevice::new();
        assert!(dev.name().contains("sovereign"));
        assert!(dev.has_f64_shaders());
        assert!(!dev.is_lost());
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn default_matches_new() {
        let dev = SovereignDevice::default();
        assert!(dev.name().contains("sovereign"));
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn dispatch_binary_without_toadstool() {
        let dev = SovereignDevice::new();
        let result = dev.dispatch_binary(&[0xDE, 0xAD], vec![], (1, 1, 1), "main");
        assert!(result.is_err());
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn buffer_alloc_stages_data() {
        let dev = SovereignDevice::new();
        let buf = dev.alloc_buffer("test", 64).unwrap();
        assert_eq!(buf.size, 64);

        dev.upload(&buf, 0, &[1, 2, 3, 4]);
        let data = dev.download(&buf, 64).unwrap();
        assert_eq!(data[..4], [1, 2, 3, 4]);
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn buffer_alloc_assigns_unique_ids() {
        let dev = SovereignDevice::new();
        let b1 = dev.alloc_buffer("a", 1024).unwrap();
        let b2 = dev.alloc_buffer("b", 2048).unwrap();
        assert_ne!(b1.id, b2.id);
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn dispatch_env_constant_is_correct() {
        assert_eq!(DISPATCH_ADDR_ENV, "BARRACUDA_DISPATCH_ADDR");
        assert_eq!(DISPATCH_CAPABILITY, "compute.dispatch");
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn detect_dispatch_addr_graceful_without_toadstool() {
        let addr = detect_dispatch_addr();
        if let Some(ref a) = addr {
            assert!(!a.is_empty(), "discovered address must be non-empty");
        }
    }

    #[test]
    fn coral_cache_lookup_returns_none_for_unknown_shader() {
        let result = SovereignDevice::try_coral_cache("nonexistent_shader_source_12345");
        assert!(result.is_none());
    }
}
