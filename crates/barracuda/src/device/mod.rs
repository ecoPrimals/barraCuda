// SPDX-License-Identifier: AGPL-3.0-only
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

use crate::error::Result;
use std::sync::Arc;

pub mod akida;
pub mod akida_executor;
pub mod async_submit;
pub mod autotune;
pub mod backend;
pub mod batched_encoder;
pub mod cache_hierarchy;
pub mod capabilities;
pub mod compute_pipeline;
pub mod coral_compiler;
#[cfg(feature = "sovereign-dispatch")]
pub mod coral_reef_device;
mod device_types;
pub mod driver_profile; // GPU driver/compiler identity + shader strategies (D-S17-002 refactor)
pub mod fma_policy; // FMA contraction policy for reproducibility (coralReef Iteration 30)
pub mod hardware_calibration; // Per-tier GPU compilation probing (hotSpring v0.6.25 absorption)
pub mod kernel_router; // Unified Math → Hardware routing (Feb 15, 2026)
pub mod latency; // LatencyModel trait + Sm70/Rdna2/AppleM/Conservative/Measured (SOVEREIGN Phase 2, Feb 2026)
pub mod pipeline_cache;
pub mod precision_brain; // Domain→tier self-routing brain (hotSpring v0.6.25 absorption)
pub mod precision_tier; // PrecisionTier, PhysicsDomain enums (hotSpring v0.6.25 absorption)
pub mod probe; // Runtime f64 exp/log capability probing (W-001 evolution)
pub mod probe_throughput; // f64 throughput ratio probing (metalForge discovery)
pub mod registry; // Physical device tracking with backend preference (Feb 16, 2026)
mod routing;
pub mod substrate;
pub mod tensor_context;
pub mod unified;
pub mod vendor; // Canonical GPU vendor ID constants (single source of truth)
pub mod warmup;
mod wgpu_backend;
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
    BufferPool, PooledBuffer, TensorContext, TensorContextStats, TensorSession,
    clear_global_contexts, get_device_context, high_capacity_limits, science_limits,
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
    CompilerKind, DeviceCapabilities, DriverKind, EigensolveStrategy, Fp64Rate, Fp64Strategy,
    GpuArch, GpuDriverProfile, Workaround, WorkloadType, optimal_workgroup_size_arch,
    workgroup_size_2d_for_arch, workgroup_size_for_arch,
};
pub use compute_pipeline::{ComputeDispatch, storage_bgl_entry, uniform_bgl_entry};
#[cfg(feature = "sovereign-dispatch")]
pub use coral_gpu;
#[cfg(feature = "sovereign-dispatch")]
pub use coral_reef_device::CoralReefDevice;
pub use kernel_router::{ComputeWorkload, KernelRouter, KernelTarget, NpuModelInfo};
pub use registry::{
    BackendInfo, DeviceCapabilities as PhysicalDeviceCapabilities, DeviceRegistry, DeviceVendor,
    PhysicalDevice, PhysicalDeviceId,
};
pub use substrate::{Substrate, SubstrateCapability, SubstrateType};
pub use unified::{Capability, Device, DeviceContext, DeviceInfo, WorkloadHint};
pub use wgpu_device::WgpuDevice;

pub use fma_policy::{FmaPolicy, domain_requires_separate_fma};
pub use hardware_calibration::HardwareCalibration;
pub use precision_brain::PrecisionBrain;
pub use precision_tier::{PhysicsDomain, PrecisionBrainAdvice, PrecisionTier};

/// Device pool for GPU operations (used by NMS and tests).
/// Always compiled so NMS can acquire a GPU device at runtime.
pub mod test_pool;

/// GPU test coordination harness — admission gate for concurrent test execution.
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
}

impl DeviceSelection {
    /// Can this device run arbitrary WGSL compute shaders?
    #[must_use]
    pub fn supports_wgsl(self) -> bool {
        matches!(self, Self::Gpu | Self::Cpu)
    }

    /// Is this device best for sparse/event-driven workloads?
    #[must_use]
    pub fn is_event_driven(self) -> bool {
        matches!(self, Self::Npu)
    }
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

/// Auto device discovery via wgpu
///
/// wgpu automatically handles:
/// - GPU (Vulkan, Metal, DX12) - preferred
/// - CPU (software rasterizer) - automatic fallback
/// - NPU/TPU (if wgpu driver available)
pub struct Auto;

impl Auto {
    /// Discover best available device (wgpu handles selection)
    /// Returns shared `WgpuDevice` from the global pool for thread-safe concurrent access.
    /// This enables parallel tests and concurrent GPU workloads without resource exhaustion.
    /// **Architecture**: Uses LazyLock-based pool (Rust 1.80+) for idiomatic lazy initialization.
    /// # Errors
    /// Returns [`Err`] if no WGPU adapter is found or device creation fails.
    #[expect(clippy::new_ret_no_self, reason = "suppressed")]
    pub async fn new() -> Result<Arc<WgpuDevice>> {
        Ok(test_pool::get_test_device().await)
    }

    /// Create a fresh device (not from pool)
    /// Use sparingly - creates a new device each call, which can exhaust GPU resources.
    /// Prefer `Auto::new()` for most cases.
    /// # Errors
    /// Returns [`Err`] if no WGPU adapter is found or device creation fails.
    pub async fn new_fresh() -> Result<WgpuDevice> {
        WgpuDevice::new().await
    }
}
