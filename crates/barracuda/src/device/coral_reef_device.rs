// SPDX-License-Identifier: AGPL-3.0-only
//! Sovereign compute backend via coralReef's `coral-gpu` crate.
//!
//! `CoralReefDevice` is the **primary** GPU backend for the sovereign pipeline.
//! It provides WGSL → native binary compilation and GPU dispatch through
//! coralReef's full pipeline, bypassing the wgpu/Vulkan/Mesa/kernel driver stack.
//!
//! # Architecture (VFIO primary)
//!
//! ```text
//! WGSL → naga → coralReef → native SASS/GFX → coral-driver → VFIO/GPFIFO → GPU
//! ```
//!
//! VFIO (via toadStool) provides exclusive device access, IOMMU hardware
//! isolation, deterministic scheduling, and zero kernel driver in the data
//! path. `dispatch_binary` is the fast path — pre-compiled native binaries
//! are submitted directly via GPFIFO.
//!
//! # Architecture (DRM fallback)
//!
//! ```text
//! WGSL → naga → coralReef → native SASS/GFX → coral-driver → DRM → GPU
//! ```
//!
//! The `GpuBackend` implementation holds a `Mutex<GpuContext>` for
//! thread-safe access. `ComputeDevice: Send + Sync` is satisfied as of
//! coralReef Iteration 26.
//!
//! # Backend maturity
//!
//! | Backend | Status | Notes |
//! |---------|--------|-------|
//! | VFIO/GPFIFO | Active design | toadStool VFIO GPU backend + `from_vfio_device` |
//! | amdgpu (DRM) | E2E verified | coralReef Phase 10 |
//! | nouveau (DRM) | Partial | Compute subchannel bound (Iter 26), validation pending |
//! | nvidia-drm | Pending | UVM integration needed |
//!
//! # Activation
//!
//! ```toml
//! [features]
//! sovereign-dispatch = ["gpu", "dep:coral-gpu"]
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
/// Wraps `coral_driver::BufferHandle` when the sovereign-dispatch feature is
/// enabled, providing an opaque GPU-side buffer reference.
#[derive(Debug, Clone, Copy)]
pub struct CoralBuffer {
    #[cfg(feature = "sovereign-dispatch")]
    handle: coral_gpu::BufferHandle,
    #[expect(dead_code, reason = "tracked for future readback size validation")]
    size: u64,
}

// CoralBuffer is Send + Sync automatically — it contains only
// BufferHandle(u32) and u64, both of which are Send + Sync.

/// Sovereign GPU compute device via coralReef.
///
/// Holds a `GpuContext` behind a `Mutex` for thread-safe `GpuBackend`
/// implementation. In compile-only mode (no hardware device), provides
/// standalone WGSL → native compilation. In dispatch mode (hardware
/// attached), provides the full alloc/upload/dispatch/readback cycle.
pub struct CoralReefDevice {
    name: Arc<str>,
    #[cfg(feature = "sovereign-dispatch")]
    ctx: std::sync::Mutex<coral_gpu::GpuContext>,
    #[cfg(feature = "sovereign-dispatch")]
    has_device: bool,
    #[cfg(feature = "sovereign-dispatch")]
    kernel_cache: std::sync::Mutex<HashMap<u64, coral_gpu::CompiledKernel>>,
}

impl std::fmt::Debug for CoralReefDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CoralReefDevice")
            .field("name", &self.name)
            .finish()
    }
}

impl CoralReefDevice {
    /// Create a compile-only `CoralReefDevice` for the given target.
    ///
    /// No hardware device is attached — `alloc`, `upload`, `dispatch`, and
    /// `download` will return errors. Use [`Self::with_auto_device`] to
    /// attach hardware for dispatch.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the feature is not enabled.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn new(target: coral_gpu::GpuTarget) -> Result<Self> {
        let ctx = coral_gpu::GpuContext::new(target).map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: context init failed: {e}"))
        })?;
        Ok(Self {
            name: Arc::from(format!("sovereign:{target:?}")),
            ctx: std::sync::Mutex::new(ctx),
            has_device: false,
            kernel_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Create a `CoralReefDevice` with auto-detected hardware.
    ///
    /// Discovers a DRM render node via coral-gpu's auto-detection and
    /// attaches it for full dispatch. Falls back to compile-only on error.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if no GPU is found or context creation fails.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn with_auto_device() -> Result<Self> {
        let ctx = coral_gpu::GpuContext::auto().map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: auto-detect failed: {e}"))
        })?;
        Ok(Self {
            name: Arc::from("sovereign:auto"),
            ctx: std::sync::Mutex::new(ctx),
            has_device: true,
            kernel_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Create a `CoralReefDevice` from a vendor/arch/driver descriptor.
    ///
    /// Used by toadStool discovery: the primal layer discovers GPUs via
    /// ecosystem IPC and passes descriptors for context creation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the descriptor is unsupported or device open fails.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn from_descriptor(vendor: &str, arch: Option<&str>, driver: Option<&str>) -> Result<Self> {
        let ctx = coral_gpu::GpuContext::from_descriptor(vendor, arch, driver).map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: descriptor failed: {e}"))
        })?;
        Ok(Self {
            name: Arc::from(format!("sovereign:{vendor}:{}", arch.unwrap_or("auto"))),
            ctx: std::sync::Mutex::new(ctx),
            has_device: true,
            kernel_cache: std::sync::Mutex::new(HashMap::new()),
        })
    }

    /// Create a `CoralReefDevice` from a VFIO device descriptor.
    ///
    /// This is the **primary** construction path for the sovereign pipeline.
    /// toadStool manages VFIO device lifecycle (IOMMU group binding, device
    /// attach/detach) and provides a `VfioGpuInfo` descriptor. barraCuda
    /// uses this to create a `GpuContext` backed by VFIO instead of DRM.
    ///
    /// # Contract
    ///
    /// - toadStool must have bound the GPU to `vfio-pci` before calling this
    /// - The IOMMU group must be viable (all devices in group bound to vfio-pci)
    /// - toadStool retains ownership of device lifecycle (unbind on drop)
    /// - barraCuda never manages VFIO bind/unbind directly
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if the VFIO device cannot be opened or is unsupported.
    /// Currently returns `Err` unconditionally — blocked on `coral-gpu` VFIO
    /// backend and toadStool's `VfioGpuInfo` type.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn from_vfio_device(pci_address: &str, vendor_id: u16, iommu_group: u32) -> Result<Self> {
        // Blocked on: coral-gpu VFIO backend + toadStool VfioGpuInfo type.
        // When unblocked, this will call coral_gpu::GpuContext::from_vfio(...)
        // with the VFIO device fd obtained from toadStool.
        Err(BarracudaError::Device(format!(
            "CoralReefDevice::from_vfio_device: VFIO backend not yet available \
             (PCI {pci_address}, vendor 0x{vendor_id:04x}, IOMMU group {iommu_group}) — \
             blocked on coral-gpu VFIO support and toadStool VfioGpuInfo type"
        )))
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

    /// Compile WGSL to a native GPU binary via coral-gpu (in-process).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if compilation fails or the feature is not enabled.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn compile_wgsl(
        &self,
        wgsl: &str,
        target: coral_gpu::GpuTarget,
    ) -> Result<coral_gpu::CompiledKernel> {
        let compile_ctx = coral_gpu::GpuContext::new(target).map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: context init failed: {e}"))
        })?;
        compile_ctx
            .compile_wgsl(wgsl)
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: compile failed: {e}")))
    }

    /// Whether this device has a hardware backend attached for dispatch.
    #[cfg(feature = "sovereign-dispatch")]
    #[must_use]
    pub fn has_dispatch(&self) -> bool {
        self.has_device
    }

    #[cfg(feature = "sovereign-dispatch")]
    fn require_device<T>(&self, method: &str) -> Result<T> {
        Err(BarracudaError::Device(format!(
            "CoralReefDevice::{method}: no hardware device attached — \
             use with_auto_device() or from_descriptor() for dispatch"
        )))
    }

    /// Check the coral compiler cache for a pre-compiled native binary.
    ///
    /// Returns `Some(binary)` if `spawn_coral_compile` has previously cached
    /// a native GPU binary for this shader source. The lookup scans all
    /// cached architectures, returning the first match.
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

    /// Dispatch a pre-compiled `CompiledKernel` with full metadata.
    ///
    /// Preferred over [`GpuBackend::dispatch_binary`] because the kernel
    /// carries GPR count, shared memory, and workgroup size from the
    /// compiler — the driver needs these for correct QMD construction.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if no hardware device is attached or dispatch fails.
    #[cfg(feature = "sovereign-dispatch")]
    pub fn dispatch_kernel(
        &self,
        kernel: &coral_gpu::CompiledKernel,
        buffers: &[CoralBuffer],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        if !self.has_device {
            return self.require_device("dispatch_kernel");
        }
        let mut ctx = self
            .ctx
            .lock()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: lock poisoned: {e}")))?;
        let buffer_handles: Vec<coral_gpu::BufferHandle> =
            buffers.iter().map(|b| b.handle).collect();
        let dims = [workgroups.0, workgroups.1, workgroups.2];
        ctx.dispatch(kernel, &buffer_handles, dims)
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: dispatch: {e}")))?;
        ctx.sync()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: sync: {e}")))
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
    fn alloc_buffer(&self, label: &str, size: u64) -> Result<CoralBuffer> {
        if !self.has_device {
            return self.require_device("alloc_buffer");
        }
        let mut ctx = self
            .ctx
            .lock()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: lock poisoned: {e}")))?;
        let handle = ctx.alloc(size).map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: alloc({label}, {size}): {e}"))
        })?;
        Ok(CoralBuffer { handle, size })
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
    fn upload(&self, buffer: &CoralBuffer, _offset: u64, data: &[u8]) {
        if let Ok(mut ctx) = self.ctx.lock() {
            let _ = ctx.upload(buffer.handle, data);
        }
    }

    #[cfg(not(feature = "sovereign-dispatch"))]
    fn upload(&self, _buffer: &CoralBuffer, _offset: u64, _data: &[u8]) {}

    #[cfg(feature = "sovereign-dispatch")]
    fn download(&self, buffer: &CoralBuffer, size: u64) -> Result<bytes::Bytes> {
        if !self.has_device {
            return self.require_device("download");
        }
        let ctx = self
            .ctx
            .lock()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: lock poisoned: {e}")))?;
        let data = ctx
            .readback(buffer.handle, size as usize)
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: readback: {e}")))?;
        Ok(bytes::Bytes::from(data))
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

        if !self.has_device {
            return self.require_device("dispatch_compute");
        }

        let mut hasher = std::hash::DefaultHasher::new();
        desc.shader_source.hash(&mut hasher);
        let key = hasher.finish();

        let mut ctx = self
            .ctx
            .lock()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: lock poisoned: {e}")))?;

        let mut cache = self.kernel_cache.lock().map_err(|e| {
            BarracudaError::Device(format!("CoralReefDevice: cache lock poisoned: {e}"))
        })?;

        let kernel = if let Some(cached) = cache.get(&key) {
            cached.clone()
        } else if let Some(cached_binary) = Self::try_coral_cache(desc.shader_source) {
            let kernel = coral_gpu::CompiledKernel {
                binary: cached_binary,
                source_hash: key,
                target: coral_gpu::GpuTarget::default(),
                gpr_count: CONSERVATIVE_GPR_COUNT,
                instr_count: 0,
                shared_mem_bytes: 0,
                barrier_count: 0,
                workgroup: DEFAULT_WORKGROUP,
            };
            tracing::debug!("sovereign cache hit — using pre-compiled native binary");
            cache.insert(key, kernel.clone());
            kernel
        } else {
            let compiled = ctx
                .compile_wgsl(desc.shader_source)
                .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: compile: {e}")))?;
            cache.insert(key, compiled.clone());
            compiled
        };

        let buffer_handles: Vec<coral_gpu::BufferHandle> =
            desc.bindings.iter().map(|b| b.buffer.handle).collect();
        let dims = [desc.workgroups.0, desc.workgroups.1, desc.workgroups.2];

        drop(cache);

        ctx.dispatch(&kernel, &buffer_handles, dims)
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: dispatch: {e}")))?;
        ctx.sync()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: sync: {e}")))
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
        use coral_gpu::{BufferHandle, CompiledKernel, GpuTarget};

        if !self.has_device {
            return self.require_device("dispatch_binary");
        }

        let kernel = CompiledKernel {
            binary: bytes::Bytes::copy_from_slice(binary),
            source_hash: 0,
            target: GpuTarget::default(),
            gpr_count: CONSERVATIVE_GPR_COUNT,
            instr_count: 0,
            shared_mem_bytes: 0,
            barrier_count: 0,
            workgroup: DEFAULT_WORKGROUP,
        };

        let mut ctx = self
            .ctx
            .lock()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: lock poisoned: {e}")))?;
        let buffer_handles: Vec<BufferHandle> = bindings.iter().map(|b| b.buffer.handle).collect();
        let dims = [workgroups.0, workgroups.1, workgroups.2];
        ctx.dispatch(&kernel, &buffer_handles, dims)
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: dispatch: {e}")))?;
        ctx.sync()
            .map_err(|e| BarracudaError::Device(format!("CoralReefDevice: sync: {e}")))
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
    fn device_creation_compile_only() {
        let target = coral_gpu::GpuTarget::Nvidia(coral_gpu::NvArch::Sm70);
        let dev = CoralReefDevice::new(target).expect("device creation");
        assert!(dev.name().contains("sovereign"));
        assert!(dev.has_f64_shaders());
        assert!(!dev.is_lost());
        assert!(!dev.has_dispatch());
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn compile_wgsl_produces_native_binary() {
        let target = coral_gpu::GpuTarget::Nvidia(coral_gpu::NvArch::Sm70);
        let dev = CoralReefDevice::new(target).expect("device creation");
        let kernel = dev
            .compile_wgsl("@compute @workgroup_size(1) fn main() {}", target)
            .expect("simple shader should compile");
        assert!(!kernel.binary.is_empty());
        assert_eq!(kernel.workgroup, [1, 1, 1]);
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn compile_physics_shader_standalone() {
        let target = coral_gpu::GpuTarget::Nvidia(coral_gpu::NvArch::Sm70);
        let dev = CoralReefDevice::new(target).expect("device creation");
        let wgsl = r"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;
            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                output[i] = input[i] * input[i] + input[i];
            }
        ";
        let kernel = dev.compile_wgsl(wgsl, target).expect("fma-like shader");
        assert!(!kernel.binary.is_empty());
        assert!(kernel.gpr_count > 0);
        assert!(kernel.instr_count > 0);
    }

    #[cfg(feature = "sovereign-dispatch")]
    #[test]
    fn compile_only_device_rejects_dispatch() {
        let target = coral_gpu::GpuTarget::Nvidia(coral_gpu::NvArch::Sm70);
        let dev = CoralReefDevice::new(target).expect("device creation");
        let result = dev.alloc_buffer("test", 1024);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("no hardware device"), "got: {msg}");
    }
}
