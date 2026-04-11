// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device module - Unified Hardware Abstraction
//!
//! **Phase 2: Unified Device Architecture**:
//! - Single Device enum for ALL hardware
//! - Automatic device selection
//! - Explicit routing when needed
//! - Flexible fallback chains
//! - Runtime capability discovery
//!
//! **Hardware Types**:
//! - CPU: Pure Rust execution
//! - GPU: WGSL shaders via wgpu
//! - NPU: Akida neuromorphic
//! - Auto: Smart selection

use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// Akida neuromorphic hardware detection and board health.
pub mod akida;
/// Akida NPU execution backend for spiking-network inference.
pub mod akida_executor;
/// Async GPU submission queue for pipelined dispatch.
pub mod async_submit;
/// Auto-tuning framework for workgroup sizes and dispatch parameters.
pub mod autotune;
/// `GpuBackend` trait abstracting wgpu, coralReef, and CPU fallback.
pub mod backend;
/// Batched command encoder for fusing multiple dispatches.
pub mod batched_encoder;
/// Multi-level GPU cache hierarchy (L1 pipeline, L2 compute, L3 global).
pub mod cache_hierarchy;
/// Runtime capability queries (f64, subgroups, max buffer size, etc.).
pub mod capabilities;
/// `ComputeDispatch` builder — the canonical GPU dispatch primitive.
pub mod compute_pipeline;
/// coralReef sovereign compiler discovery and gRPC bridge.
pub mod coral_compiler;
mod device_types;
/// GPU driver/compiler identity and shader strategy selection.
pub mod driver_profile;
/// FMA contraction policy for reproducible floating-point results.
pub mod fma_policy;
/// Per-tier GPU compilation probing (hotSpring v0.6.25 absorption).
pub mod hardware_calibration;
/// Unified Math-to-Hardware routing (operation → best device).
pub mod kernel_router;
/// Latency models for dispatch scheduling (Sm70, Rdna2, AppleM, Conservative).
pub mod latency;
/// Global GPU pipeline cache — deduplicates compiled shader modules.
pub mod pipeline_cache;
/// Domain-aware precision self-routing brain.
pub mod precision_brain;
/// `PrecisionTier` and `PhysicsDomain` enums.
pub mod precision_tier;
/// Runtime f64 exp/log capability probing.
pub mod probe;
/// f64 throughput ratio probing (native vs emulated performance).
pub mod probe_throughput;
/// Physical device registry with backend preference ordering.
pub mod registry;
mod routing;
/// Unified `Device` enum routing across CPU, GPU, NPU, and Auto.
pub mod silicon_profile;
/// Sovereign device implementation (capability-based IPC dispatch).
#[cfg(feature = "sovereign-dispatch")]
pub mod sovereign_device;
/// GPU memory substrate — allocation, lifetime, and residency tracking.
pub mod substrate;
/// Zero-overhead tensor context with buffer pooling.
pub mod tensor_context;
pub mod unified;
/// Canonical GPU vendor ID constants (single source of truth).
pub mod vendor;
/// Device warm-up (mise en place) — pre-compile pipelines before first use.
pub mod warmup;
mod wgpu_backend;
/// `WgpuDevice` — the primary GPU device implementation via wgpu/WebGPU.
pub mod wgpu_device;

// Re-export auto-tuning types
pub use autotune::{AutoTuner, GLOBAL_TUNER, GpuCalibration};

// Re-export warmup (mise en place)
pub use warmup::{
    WarmupConfig, WarmupOp, WarmupResult, WarmupWorkloadHint, warmup_device, warmup_pool,
};

// Re-export pipeline cache (for testing cache clearing)
pub use pipeline_cache::clear_global_cache;

// Re-export tensor context (zero-overhead Tensor operations)
pub use tensor_context::{
    BatchGuard, BufferPool, PooledBuffer, TensorContext, TensorContextStats, clear_global_contexts,
    get_device_context, high_capacity_limits, science_limits,
};

pub use akida::{AkidaBoard, AkidaCapabilities, BoardHealth, detect_akida_boards};
pub use akida_executor::{AkidaExecutor, NeuromorphicComparison};
pub use async_submit::{AsyncReadback, AsyncSubmitter};
pub use backend::{BufferBinding, DispatchDescriptor, GpuBackend};
pub use batched_encoder::BatchedEncoder;
pub use cache_hierarchy::{
    CacheAwareTiler, CacheLevel, CacheResidency, MainMemory, SubstrateMemoryHierarchy, TileConfig,
};
pub use capabilities::{
    DeviceCapabilities, EigensolveStrategy, Fp64Rate, Fp64Strategy, PrecisionRoutingAdvice,
    WorkloadType, optimal_workgroup_size_arch, workgroup_size_2d_for_arch, workgroup_size_for_arch,
};
pub use compute_pipeline::{
    BatchedComputeDispatch, ComputeDispatch, storage_bgl_entry, uniform_bgl_entry,
};
pub use kernel_router::{ComputeWorkload, KernelRouter, KernelTarget, NpuModelInfo};
pub use registry::{
    BackendInfo, DeviceCapabilities as PhysicalDeviceCapabilities, DeviceRegistry, DeviceVendor,
    PhysicalDevice, PhysicalDeviceId,
};
#[cfg(feature = "sovereign-dispatch")]
pub use sovereign_device::SovereignDevice;
pub use substrate::{Substrate, SubstrateCapability, SubstrateType};
pub use unified::{Capability, Device, DeviceContext, DeviceInfo, WorkloadHint};
pub use wgpu_device::WgpuDevice;

// Re-export the tiered discovery result type.
// `DiscoveredDevice` is the return type of `Auto::new()` after BC-07.

pub use fma_policy::{FmaPolicy, domain_requires_separate_fma};
pub use hardware_calibration::HardwareCalibration;
pub use precision_brain::PrecisionBrain;
pub use precision_tier::{PhysicsDomain, PrecisionBrainAdvice, PrecisionTier};

/// Device pool for GPU operations (used by NMS and tests).
/// Always compiled so NMS can acquire a GPU device at runtime.
pub mod test_pool;

/// GPU test coordination harness — admission gate for concurrent test execution.
///
/// Always compiled so integration tests can use `gpu_section` and `with_coral`.
pub mod test_harness;

/// Which hardware to target and how
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSelection {
    /// GPU via WGPU/WGSL - runs any operation, hardware-accelerated
    Gpu,
    /// CPU via WGPU software rasterizer - runs any operation, slower
    Cpu,
    /// NPU via Akida driver - runs pre-compiled SNN models only
    Npu,
    /// Sovereign GPU via IPC dispatch (no wgpu/Vulkan/Metal)
    Sovereign,
}

impl DeviceSelection {
    /// Can this device run arbitrary WGSL compute shaders?
    #[must_use]
    pub fn supports_wgsl(self) -> bool {
        matches!(self, Self::Gpu | Self::Cpu | Self::Sovereign)
    }

    /// Is this device best for sparse/event-driven workloads?
    #[must_use]
    pub fn is_event_driven(self) -> bool {
        matches!(self, Self::Npu)
    }

    /// Is this a sovereign dispatch path (pure Rust, no Vulkan)?
    #[must_use]
    pub fn is_sovereign(self) -> bool {
        matches!(self, Self::Sovereign)
    }
}

/// Check whether sovereign dispatch is available at runtime.
///
/// Returns `true` when the `sovereign-dispatch` feature is enabled **and**
/// `SovereignDevice::with_auto_device()` finds a dispatchable GPU via
/// capability-based IPC discovery.
/// This is a runtime probe — call once at startup, cache the result.
#[must_use]
pub fn sovereign_available() -> bool {
    #[cfg(feature = "sovereign-dispatch")]
    {
        sovereign_device::SovereignDevice::with_auto_device()
            .map(|d| d.has_dispatch())
            .unwrap_or(false)
    }
    #[cfg(not(feature = "sovereign-dispatch"))]
    false
}

/// What kind of work needs to be done (hardware routing level)
///
/// Used by the kernel router and device selection to pick optimal hardware.
/// Each variant maps to different backend preferences (e.g. NPU for spiking, GPU for dense matmul).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareWorkload {
    /// General tensor ops (matmul, elementwise) — GPU preferred.
    TensorOps,
    /// Dense neural networks (MLP, attention) — GPU preferred.
    NeuralNetwork,
    /// Spiking neural networks — NPU preferred when available.
    SpikingNetwork,
    /// Reservoir computing / ESN — GPU or CPU depending on size.
    ReservoirComputing,
    /// Sequence alignment, k-mer analysis — GPU for batch, CPU for small.
    Genomics,
    /// Phylogenetics, HMM, Smith-Waterman — GPU batch preferred.
    Bioinformatics,
    /// PDE/ODE, FFT, sparse solvers — GPU f64 preferred.
    ScientificCompute,
    /// FHE polynomial ops (NTT, key switch) — GPU preferred.
    HomomorphicEncryption,
}

/// Discovered compute device — the result of [`Auto::new()`].
///
/// Wraps the best available compute backend so callers can dispatch
/// without knowing which tier was selected.  Use [`wgpu_device()`](Self::wgpu_device)
/// when a concrete `WgpuDevice` is required (tensor creation, GPU readback).
#[derive(Debug, Clone)]
pub enum DiscoveredDevice {
    /// Tiers 1–2: local GPU or CPU software rasterizer via wgpu.
    Wgpu(Arc<WgpuDevice>),
    /// Tier 3: GPU compute via IPC to coralReef (compile) + toadStool (dispatch).
    #[cfg(feature = "sovereign-dispatch")]
    Sovereign(Arc<sovereign_device::SovereignDevice>),
}

impl DiscoveredDevice {
    /// Human-readable name of the underlying device.
    #[must_use]
    pub fn name(&self) -> String {
        match self {
            Self::Wgpu(d) => d.adapter_info().name.clone(),
            #[cfg(feature = "sovereign-dispatch")]
            Self::Sovereign(d) => {
                use backend::GpuBackend;
                d.name().to_owned()
            }
        }
    }

    /// Whether the device supports native f64 shader builtins.
    #[must_use]
    pub fn has_f64_shaders(&self) -> bool {
        match self {
            Self::Wgpu(d) => d.has_f64_shaders(),
            #[cfg(feature = "sovereign-dispatch")]
            Self::Sovereign(d) => {
                use backend::GpuBackend;
                d.has_f64_shaders()
            }
        }
    }

    /// Extract the inner `WgpuDevice` if this is a local wgpu backend.
    ///
    /// Returns `None` for sovereign IPC — callers that need local tensor
    /// buffers should degrade gracefully or forward via IPC.
    #[must_use]
    pub fn wgpu_device(&self) -> Option<&Arc<WgpuDevice>> {
        match self {
            Self::Wgpu(d) => Some(d),
            #[cfg(feature = "sovereign-dispatch")]
            Self::Sovereign(_) => None,
        }
    }

    /// Whether this is the sovereign IPC path (no local GPU).
    #[must_use]
    pub fn is_sovereign(&self) -> bool {
        match self {
            Self::Wgpu(_) => false,
            #[cfg(feature = "sovereign-dispatch")]
            Self::Sovereign(_) => true,
        }
    }

    /// Extract the wgpu device, returning an error for sovereign.
    ///
    /// Convenience for call sites that cannot operate without local wgpu buffers.
    /// # Errors
    /// Returns [`Err`] when the device is sovereign IPC.
    pub fn require_wgpu(self) -> Result<Arc<WgpuDevice>> {
        match self {
            Self::Wgpu(d) => Ok(d),
            #[cfg(feature = "sovereign-dispatch")]
            Self::Sovereign(_) => Err(BarracudaError::device(
                "Operation requires a local wgpu device; sovereign IPC cannot provide local buffers",
            )),
        }
    }
}

/// Auto device discovery — tiered fallback across all backends
///
/// Fallback chain (BC-07):
/// 1. GPU via wgpu (Vulkan, Metal, DX12) — fastest
/// 2. CPU software rasterizer via wgpu — universal but slow
/// 3. `SovereignDevice` via IPC to coralReef+toadStool — GPU via IPC
///    (requires `sovereign-dispatch` feature and live peers)
/// 4. `Err` — callers degrade gracefully (cpu-shader still available)
pub struct Auto;

impl Auto {
    /// Discover the best available compute device.
    ///
    /// Tries adapters in order:
    /// 1. GPU (wgpu `HighPerformance`)
    /// 2. CPU software rasterizer (wgpu `LowPower`)
    /// 3. Sovereign IPC (coralReef + toadStool) when `sovereign-dispatch`
    ///    feature is enabled and a `compute.dispatch` endpoint is discoverable
    ///
    /// Returns [`DiscoveredDevice`] wrapping whichever tier succeeded.
    /// Returns `Err` only if **all** tiers fail — callers (e.g.
    /// `BarraCudaPrimal::start`) can still degrade to cpu-shader.
    ///
    /// For tests, prefer `test_pool::get_test_device()` which manages a shared
    /// pool with concurrency limits.
    /// # Errors
    /// Returns [`Err`] if no wgpu adapter is found and sovereign dispatch is
    /// unavailable.
    #[expect(
        clippy::new_ret_no_self,
        reason = "Auto is a discovery namespace, not a constructible type"
    )]
    pub async fn new() -> Result<DiscoveredDevice> {
        if let Ok(dev) = WgpuDevice::new().await {
            return Ok(DiscoveredDevice::Wgpu(Arc::new(dev)));
        }
        if let Ok(dev) = WgpuDevice::new_cpu_relaxed().await {
            tracing::info!("GPU unavailable, using CPU software rasterizer");
            return Ok(DiscoveredDevice::Wgpu(Arc::new(dev)));
        }
        #[cfg(feature = "sovereign-dispatch")]
        if sovereign_available() {
            if let Ok(sov) = sovereign_device::SovereignDevice::with_auto_device() {
                tracing::info!(
                    "wgpu unavailable, using sovereign IPC dispatch via coralReef+toadStool"
                );
                return Ok(DiscoveredDevice::Sovereign(Arc::new(sov)));
            }
        }
        Err(BarracudaError::device(
            "No compute device available (GPU, CPU software rasterizer, \
             and sovereign IPC all unavailable)",
        ))
    }

    /// Convenience: discover the best **wgpu** device (tiers 1–2 only).
    ///
    /// Identical to `Auto::new()` but returns `Arc<WgpuDevice>` directly,
    /// skipping sovereign IPC. Use this when you need local tensor buffers
    /// (e.g. `Tensor::from_vec`, `TensorSession`, tests).
    /// # Errors
    /// Returns [`Err`] if no wgpu adapter (GPU or CPU software) is found.
    pub async fn new_wgpu() -> Result<Arc<WgpuDevice>> {
        if let Ok(dev) = WgpuDevice::new().await {
            return Ok(Arc::new(dev));
        }
        if let Ok(dev) = WgpuDevice::new_cpu_relaxed().await {
            tracing::info!("GPU unavailable, using CPU software rasterizer");
            return Ok(Arc::new(dev));
        }
        Err(BarracudaError::device(
            "No wgpu device available (GPU and CPU software rasterizer both unavailable)",
        ))
    }

    /// Create a fresh wgpu device (not from pool).
    ///
    /// Use sparingly — creates a new device each call, which can exhaust GPU resources.
    /// Prefer `Auto::new()` for most cases.
    /// # Errors
    /// Returns [`Err`] if no wgpu adapter is found or device creation fails.
    pub async fn new_fresh() -> Result<WgpuDevice> {
        WgpuDevice::new().await
    }
}
