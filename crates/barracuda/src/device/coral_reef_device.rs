// SPDX-License-Identifier: AGPL-3.0-only
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
//! | IPC dispatch (toadStool) | In progress | `compute.dispatch.submit` (S152) |
//! | DRM (nouveau/amdgpu) | E2E verified | Titan V + RTX 3090 proven via coralReef |
//! | VFIO/GPFIFO | 6/7 tests pass | PFIFO channel init pending in coralReef |
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

/// Conservative default GPR count for pre-compiled binary dispatch.
///
/// Used when the compiler cache returns a binary without metadata.
/// 128 is safe for all current targets (SM70+, GFX1030+) — the driver
/// may over-allocate registers but will not fail.
#[cfg(feature = "sovereign-dispatch")]
const CONSERVATIVE_GPR_COUNT: u32 = 128;

/// Default workgroup size for pre-compiled binary dispatch.
#[cfg(feature = "sovereign-dispatch")]
const DEFAULT_WORKGROUP: [u32; 3] = [64, 1, 1];

/// Architectures to scan when looking up pre-compiled native binaries
/// from the coral compiler cache.
const CORAL_CACHE_ARCHITECTURES: &[&str] = &[
    "sm_70", "sm_75", "sm_80", "sm_86", "sm_89", "gfx1030", "gfx1100",
];

/// Buffer handle for the sovereign compute path.
///
/// Wraps a dispatch-side buffer identifier returned from toadStool's
/// `compute.dispatch.submit` IPC. The `id` is opaque to barraCuda —
/// toadStool manages the actual GPU memory.
#[derive(Debug, Clone, Copy)]
pub struct CoralBuffer {
    #[cfg(feature = "sovereign-dispatch")]
    id: u64,
    #[expect(dead_code, reason = "tracked for future readback size validation")]
    size: u64,
}

/// Sovereign GPU compute device via IPC to coralReef + toadStool.
///
/// Compilation flows through the existing `coral_compiler/` JSON-RPC
/// client to coralReef. Dispatch will flow through toadStool's
/// `compute.dispatch.submit` endpoint. Both are runtime IPC — no
/// compile-time coupling to any primal crate.
pub struct CoralReefDevice {
    name: Arc<str>,
    #[cfg(feature = "sovereign-dispatch")]
    dispatch_available: bool,
    #[cfg(feature = "sovereign-dispatch")]
    binary_cache: std::sync::Mutex<HashMap<u64, CachedBinary>>,
}

/// A compiled native GPU binary cached from coralReef IPC compilation.
#[cfg(feature = "sovereign-dispatch")]
#[derive(Clone)]
struct CachedBinary {
    binary: bytes::Bytes,
    gpr_count: u32,
    workgroup: [u32; 3],
}

impl std::fmt::Debug for CoralReefDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoralReefDevice")
            .field("name", &self.name)
            .finish()
    }
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
        let dispatch_available = crate::device::coral_compiler::is_coral_available();
        Self {
            name: Arc::from("sovereign-ipc (coralReef compile + toadStool dispatch)"),
            dispatch_available,
            binary_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }

    /// Create a `CoralReefDevice` (feature not enabled).
    #[cfg(not(feature = "sovereign-dispatch"))]
    pub fn new_disabled() -> Result<Self> {
        Err(BarracudaError::Device(
            "CoralReefDevice: enable the 'sovereign-dispatch' feature — \
             cargo build --features sovereign-dispatch"
                .into(),
        ))
    }

    /// Whether coralReef is available for compilation via IPC.
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn has_compiler(&self) -> bool {
        self.dispatch_available
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
        // toadStool dispatch IPC will manage GPU buffers.
        // For now, track locally — buffer IDs will be assigned by toadStool
        // once compute.dispatch.submit is wired.
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Ok(CoralBuffer {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            size,
        })
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
    fn upload(&self, _buffer: &CoralBuffer, _offset: u64, _data: &[u8]) {
        // Data upload will be handled by toadStool compute.dispatch.submit
        // as part of the dispatch payload. Buffer contents are staged locally
        // and sent with the dispatch request.
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn upload(&self, _buffer: &CoralBuffer, _offset: u64, _data: &[u8]) {}

    #[cfg(feature = "sovereign-dispatch")]
    fn download(&self, _buffer: &CoralBuffer, _size: u64) -> Result<bytes::Bytes> {
        // Readback will be handled by toadStool compute.dispatch.result
        Err(BarracudaError::Device(
            "CoralReefDevice: dispatch readback pending toadStool \
             compute.dispatch.result IPC wiring"
                .into(),
        ))
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn download(&self, _buffer: &CoralBuffer, _size: u64) -> Result<bytes::Bytes> {
        Err(BarracudaError::Device(
            "sovereign-dispatch not enabled".into(),
        ))
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn dispatch_compute(&self, desc: DispatchDescriptor<'_, Self>) -> Result<()> {
        use std::hash::{Hash, Hasher};

        let mut hasher = std::hash::DefaultHasher::new();
        desc.shader_source.hash(&mut hasher);
        let key = hasher.finish();

        let mut cache = self.binary_cache.lock().map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: cache lock poisoned: {e}"))
        })?;

        if cache.contains_key(&key) {
            tracing::debug!("sovereign cache hit — using pre-compiled native binary");
        } else if let Some(cached_binary) = Self::try_coral_cache(desc.shader_source) {
            cache.insert(
                key,
                CachedBinary {
                    binary: cached_binary,
                    gpr_count: CONSERVATIVE_GPR_COUNT,
                    workgroup: DEFAULT_WORKGROUP,
                },
            );
            tracing::debug!("coral compiler cache hit — using pre-compiled native binary");
        } else {
            // Compile via coralReef IPC (shader.compile.wgsl)
            // The coral_compiler module handles discovery and compilation.
            // For now, return an error — full IPC dispatch wiring is next.
            return Err(BarracudaError::Device(
                "CoralReefDevice: live IPC compilation + dispatch pending — \
                 use WgpuDevice for now, or pre-compile via spawn_coral_compile()"
                    .into(),
            ));
        }

        // Dispatch via toadStool IPC (compute.dispatch.submit)
        // This will send the compiled binary + buffer data to toadStool,
        // which routes to the best available GPU (VFIO/DRM).
        Err(BarracudaError::Device(
            "CoralReefDevice: toadStool compute.dispatch.submit IPC wiring in progress".into(),
        ))
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
        _bindings: Vec<BufferBinding<'_, Self>>,
        _workgroups: (u32, u32, u32),
        _entry_point: &str,
    ) -> Result<()> {
        let _ = binary;
        // dispatch_binary sends a pre-compiled native GPU binary to
        // toadStool's compute.dispatch.submit for execution.
        Err(BarracudaError::Device(
            "CoralReefDevice: toadStool compute.dispatch.submit IPC wiring in progress".into(),
        ))
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
    fn dispatch_returns_pending_error() {
        let dev = CoralReefDevice::new();
        let result = dev.dispatch_binary(&[], vec![], (1, 1, 1), "main");
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("IPC wiring"), "got: {msg}");
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn buffer_alloc_assigns_ids() {
        let dev = CoralReefDevice::new();
        let b1 = dev.alloc_buffer("a", 1024).unwrap();
        let b2 = dev.alloc_buffer("b", 2048).unwrap();
        assert_ne!(b1.id, b2.id);
        assert_eq!(b1.size, 1024);
        assert_eq!(b2.size, 2048);
    }
}
