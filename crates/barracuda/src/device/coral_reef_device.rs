// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign compute backend via IPC to coralReef + toadStool.
//!
//! `CoralReefDevice` is the **primary** GPU backend for the sovereign pipeline.
//! It uses JSON-RPC IPC to communicate with the ecosystem at runtime:
//!
//! - **coralReef** compiles WGSL → native GPU binaries (`shader.compile.wgsl`)
//! - **toadStool** dispatches binaries to hardware (`compute.dispatch.submit`)
//!
//! No compile-time coupling to coral-gpu or any other primal crate.
//! barraCuda is one math library for any deployment — the GPU backend is
//! determined entirely at runtime by which primals are available.
//!
//! # Architecture (IPC-first)
//!
//! ```text
//! barraCuda (WGSL math)
//!   → [JSON-RPC] coralReef (WGSL → native SASS/GFX binary)
//!     → [JSON-RPC] toadStool (dispatch binary → VFIO/DRM → GPU)
//!       → GPU hardware (any PCIe GPU — NVIDIA, AMD, Intel)
//! ```
//!
//! barraCuda never touches VFIO, DRM, or GPU hardware directly.
//! toadStool owns all hardware lifecycle (VFIO bind/unbind, DMA, thermal).
//! coralReef owns all compilation (naga → native binary, f64 lowering).
//!
//! # Backend maturity
//!
//! | Path | Status | Notes |
//! |------|--------|-------|
//! | IPC compile (coralReef) | Done | `shader.compile.wgsl` via `coral_compiler/` |
//! | IPC dispatch (toadStool) | Wired | `compute.dispatch.submit` (S152); readback pending |
//! | DRM (nouveau/amdgpu) | E2E verified | Titan V + RTX 3090 proven via coralReef |
//! | VFIO/GPFIFO | Fix applied | USERD_TARGET + INST_TARGET fix (Iter 44); hw revalidation pending |
//!
//! # Activation
//!
//! ```toml
//! [features]
//! sovereign-dispatch = ["gpu"]
//! ```

use super::backend::{BufferBinding, DispatchDescriptor, GpuBackend};
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

/// Architectures to scan when looking up pre-compiled native binaries
/// from the coral compiler cache.
const CORAL_CACHE_ARCHITECTURES: &[&str] = &[
    "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "sm_100", "gfx1030", "gfx1100",
];

/// Capability string used for toadStool dispatch discovery.
const DISPATCH_CAPABILITY: &str = "compute.dispatch";

/// Environment variable for explicit toadStool dispatch endpoint.
const TOADSTOOL_ADDR_ENV: &str = "BARRACUDA_DISPATCH_ADDR";

/// Default discovery directory fallback when `ECOPRIMALS_DISCOVERY_DIR` is not set.
const DEFAULT_ECOPRIMALS_DISCOVERY_DIR: &str = "ecoPrimals";

/// Canonical discovery subdirectory name.
const DISCOVERY_SUBDIR: &str = "discovery";

/// Serialisable buffer binding descriptor for IPC dispatch to toadStool.
///
/// Carries the buffer identity and access mode across the JSON-RPC boundary.
/// toadStool uses `buffer_id` to resolve staged GPU memory, `size` for
/// validation, and `read_only` to set appropriate access flags.
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
/// during staging; toadStool maps it to actual GPU memory on dispatch.
#[derive(Debug, Clone, Copy)]
pub struct CoralBuffer {
    #[cfg(feature = "sovereign-dispatch")]
    id: u64,
    size: u64,
}

/// Sovereign GPU compute device via IPC to coralReef + toadStool.
///
/// Compilation flows through the existing `coral_compiler/` JSON-RPC
/// client to coralReef. Dispatch flows through toadStool's
/// `compute.dispatch.submit` endpoint. Both are runtime IPC — no
/// compile-time coupling to any primal crate.
pub struct CoralReefDevice {
    name: Arc<str>,
    #[cfg(feature = "sovereign-dispatch")]
    compiler_available: bool,
    #[cfg(feature = "sovereign-dispatch")]
    dispatch_addr: Option<String>,
    #[cfg(feature = "sovereign-dispatch")]
    binary_cache: std::sync::Mutex<HashMap<u64, CachedBinary>>,
    #[cfg(feature = "sovereign-dispatch")]
    staged_buffers: std::sync::Mutex<HashMap<u64, bytes::BytesMut>>,
}

/// A compiled native GPU binary cached from coralReef IPC compilation.
///
/// `gpr_count` and `workgroup` are tracked for toadStool dispatch metadata.
/// They will be sent as part of the `compute.dispatch.submit` payload once
/// toadStool accepts kernel metadata alongside the binary.
#[cfg(feature = "sovereign-dispatch")]
#[derive(Clone)]
struct CachedBinary {
    binary: bytes::Bytes,
    #[expect(dead_code, reason = "sent to toadStool with dispatch metadata (P1)")]
    gpr_count: u32,
    #[expect(dead_code, reason = "sent to toadStool with dispatch metadata (P1)")]
    workgroup: [u32; 3],
}

impl std::fmt::Debug for CoralReefDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoralReefDevice")
            .field("name", &self.name)
            .finish()
    }
}

/// Detect whether a toadStool dispatch endpoint is available at runtime.
///
/// Discovery chain (first match wins):
/// 1. `BARRACUDA_DISPATCH_ADDR` env var
/// 2. Capability scan of `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
///    `"compute.dispatch"` in the `provides` or `capabilities` array
#[cfg(feature = "sovereign-dispatch")]
fn detect_dispatch_addr() -> Option<String> {
    if let Ok(addr) = std::env::var(TOADSTOOL_ADDR_ENV) {
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

impl CoralReefDevice {
    /// Create a `CoralReefDevice` that compiles via coralReef IPC and
    /// dispatches via toadStool IPC.
    ///
    /// Both primals are discovered at runtime via capability-based discovery.
    /// If neither is available, the device can still serve as a compile-only
    /// backend (using the coral compiler cache for pre-compiled binaries).
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn new() -> Self {
        let compiler_available = crate::device::coral_compiler::is_coral_available();
        let dispatch_addr = detect_dispatch_addr();
        if let Some(ref addr) = dispatch_addr {
            tracing::info!(addr = %addr, "discovered toadStool dispatch endpoint");
        }
        Self {
            name: Arc::from("sovereign-ipc (coralReef compile + toadStool dispatch)"),
            compiler_available,
            dispatch_addr,
            binary_cache: std::sync::Mutex::new(HashMap::new()),
            staged_buffers: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create a `CoralReefDevice` (feature not enabled).
    ///
    /// # Errors
    ///
    /// Always returns `Err` when the `sovereign-dispatch` feature is disabled.
    #[cfg(not(feature = "sovereign-dispatch"))]
    pub fn new_disabled() -> Result<Self> {
        Err(BarracudaError::Device(
            "CoralReefDevice: enable the 'sovereign-dispatch' feature — \
             cargo build --features sovereign-dispatch"
                .into(),
        ))
    }

    /// Attempt to create a `CoralReefDevice` with auto-detected hardware.
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

    /// Attempt to create a `CoralReefDevice` (feature not enabled).
    ///
    /// # Errors
    ///
    /// Always returns `Err` when the `sovereign-dispatch` feature is disabled.
    #[cfg(not(feature = "sovereign-dispatch"))]
    pub fn with_auto_device() -> Result<Self> {
        Err(BarracudaError::Device(
            "CoralReefDevice: enable the 'sovereign-dispatch' feature".into(),
        ))
    }

    /// Whether coralReef is available for compilation via IPC.
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn has_compiler(&self) -> bool {
        self.compiler_available
    }

    /// Whether toadStool dispatch is available for this device.
    ///
    /// Returns `true` when a toadStool endpoint advertising
    /// `compute.dispatch` was discovered at construction time.
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
    fn try_coral_cache(shader_source: &str) -> Option<bytes::Bytes> {
        use crate::device::coral_compiler::{cached_native_binary, shader_hash};
        let hash = shader_hash(shader_source);
        for arch in CORAL_CACHE_ARCHITECTURES {
            if let Some(binary) = cached_native_binary(&hash, arch) {
                return Some(binary.binary);
            }
        }
        None
    }

    /// Submit a dispatch request to toadStool via JSON-RPC.
    ///
    /// Sends the compiled native binary, workgroup dimensions, buffer binding
    /// descriptors (IDs + access mode), and hardware routing hint. toadStool
    /// maps buffer IDs to GPU memory and routes to the appropriate hardware unit.
    #[cfg(feature = "sovereign-dispatch")]
    fn submit_to_toadstool(
        &self,
        binary: &[u8],
        workgroups: (u32, u32, u32),
        bindings: &[IpcBufferBinding],
        hardware_hint: super::backend::HardwareHint,
    ) -> Result<()> {
        let Some(ref addr) = self.dispatch_addr else {
            return Err(BarracudaError::Device(
                "CoralReefDevice: no toadStool dispatch endpoint discovered — \
                 ensure toadStool is running and advertising compute.dispatch"
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
        });

        let addr = addr.clone();
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            return Err(BarracudaError::Device(
                "CoralReefDevice: no tokio runtime available for IPC dispatch".into(),
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
                            "CoralReefDevice: toadStool connect to {host_port}: {e}"
                        ))
                    })?;

                tokio::io::AsyncWriteExt::write_all(&mut stream, http_request.as_bytes())
                    .await
                    .map_err(|e| {
                        BarracudaError::Device(format!("CoralReefDevice: dispatch write: {e}"))
                    })?;

                let mut response_buf = Vec::new();
                tokio::io::AsyncReadExt::read_to_end(&mut stream, &mut response_buf)
                    .await
                    .map_err(|e| {
                        BarracudaError::Device(format!("CoralReefDevice: dispatch read: {e}"))
                    })?;

                let response_str = String::from_utf8_lossy(&response_buf);
                let json_start = response_str.find('{').ok_or_else(|| {
                    BarracudaError::Device(
                        "CoralReefDevice: no JSON body in dispatch response".into(),
                    )
                })?;

                let rpc_response: serde_json::Value =
                    serde_json::from_str(&response_str[json_start..]).map_err(|e| {
                        BarracudaError::Device(format!("CoralReefDevice: parse response: {e}"))
                    })?;

                if let Some(error) = rpc_response.get("error") {
                    let msg = error
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown error");
                    return Err(BarracudaError::Device(format!(
                        "CoralReefDevice: toadStool dispatch error: {msg}"
                    )));
                }

                tracing::debug!(
                    addr = %host_port,
                    "sovereign dispatch submitted to toadStool"
                );
                Ok(())
            })
        })
    }
}

#[cfg(feature = "sovereign-dispatch")]
impl Default for CoralReefDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for CoralReefDevice {
    type Buffer = CoralBuffer;

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
    fn alloc_buffer(&self, _label: &str, size: u64) -> Result<CoralBuffer> {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if let Ok(mut staged) = self.staged_buffers.lock() {
            staged.insert(id, bytes::BytesMut::zeroed(size as usize));
        }
        Ok(CoralBuffer { id, size })
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn alloc_buffer(&self, _label: &str, _size: u64) -> Result<CoralBuffer> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn alloc_buffer_init(&self, label: &str, contents: &[u8]) -> Result<CoralBuffer> {
        let buf = self.alloc_buffer(label, contents.len() as u64)?;
        self.upload(&buf, 0, contents);
        Ok(buf)
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn alloc_buffer_init(&self, _label: &str, _contents: &[u8]) -> Result<CoralBuffer> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn alloc_uniform(&self, label: &str, contents: &[u8]) -> Result<CoralBuffer> {
        self.alloc_buffer_init(label, contents)
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn alloc_uniform(&self, _label: &str, _contents: &[u8]) -> Result<CoralBuffer> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn upload(&self, buffer: &CoralBuffer, offset: u64, data: &[u8]) {
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
    fn upload(&self, _buffer: &CoralBuffer, _offset: u64, _data: &[u8]) {}

    #[cfg(feature = "sovereign-dispatch")]
    fn download(&self, buffer: &CoralBuffer, _size: u64) -> Result<bytes::Bytes> {
        let staged = self.staged_buffers.lock().map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: staged lock poisoned: {e}"))
        })?;
        staged
            .get(&buffer.id)
            .map(|b| bytes::Bytes::copy_from_slice(b))
            .ok_or_else(|| {
                BarracudaError::Device(format!(
                    "CoralReefDevice: buffer {} not found in staging",
                    buffer.id
                ))
            })
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn download(&self, _buffer: &CoralBuffer, _size: u64) -> Result<bytes::Bytes> {
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
            BarracudaError::Device(format!("CoralReefDevice: cache lock poisoned: {e}"))
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
                        "CoralReefDevice: no cached binary and live IPC compilation \
                         not yet implemented — pre-compile via spawn_coral_compile()"
                            .into(),
                    ));
                }
            }
        };

        let binary = cached.binary.clone();
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

        self.submit_to_toadstool(&binary, desc.workgroups, &ipc_bindings, desc.hardware_hint)
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

        self.submit_to_toadstool(
            binary,
            workgroups,
            &ipc_bindings,
            super::backend::HardwareHint::Compute,
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
        let dev = CoralReefDevice::new();
        assert!(dev.name().contains("sovereign"));
        assert!(dev.has_f64_shaders());
        assert!(!dev.is_lost());
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn default_matches_new() {
        let dev = CoralReefDevice::default();
        assert!(dev.name().contains("sovereign"));
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn dispatch_binary_without_toadstool() {
        let dev = CoralReefDevice::new();
        let result = dev.dispatch_binary(&[0xDE, 0xAD], vec![], (1, 1, 1), "main");
        assert!(result.is_err());
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn buffer_alloc_stages_data() {
        let dev = CoralReefDevice::new();
        let buf = dev.alloc_buffer("test", 64).unwrap();
        assert_eq!(buf.size, 64);

        dev.upload(&buf, 0, &[1, 2, 3, 4]);
        let data = dev.download(&buf, 64).unwrap();
        assert_eq!(data[..4], [1, 2, 3, 4]);
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn buffer_alloc_assigns_unique_ids() {
        let dev = CoralReefDevice::new();
        let b1 = dev.alloc_buffer("a", 1024).unwrap();
        let b2 = dev.alloc_buffer("b", 2048).unwrap();
        assert_ne!(b1.id, b2.id);
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn dispatch_env_constant_is_correct() {
        assert_eq!(TOADSTOOL_ADDR_ENV, "BARRACUDA_DISPATCH_ADDR");
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
    fn coral_cache_architectures_non_empty() {
        assert!(!CORAL_CACHE_ARCHITECTURES.is_empty());
        for arch in CORAL_CACHE_ARCHITECTURES {
            assert!(
                arch.starts_with("sm_") || arch.starts_with("gfx"),
                "unexpected arch format: {arch}"
            );
        }
    }
}
