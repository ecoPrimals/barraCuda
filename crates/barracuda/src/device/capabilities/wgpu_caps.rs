// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device Capability Detection — Runtime Hardware Limits (wgpu).
//!
//! Answers "what can this wgpu device do?" by querying the adapter at
//! construction time and providing typed, zero-hardcoded accessors.

use crate::device::WgpuDevice;
use crate::device::driver_profile::GpuArch;
use crate::device::probe::F64BuiltinCapabilities;
use crate::device::vendor::{VENDOR_AMD, VENDOR_INTEL, VENDOR_NVIDIA};

/// Minimum buffer size (bytes) for FHE workloads — 16K degree polynomial estimate.
pub const FHE_MIN_BUFFER_SIZE: u64 = 256 * 1024;

/// Minimum invocations per workgroup to consider device "high performance".
pub const HIGH_PERFORMANCE_MIN_INVOCATIONS: u32 = 1024;

/// Bytes per megabyte (for display formatting).
const BYTES_PER_MB: u64 = 1024 * 1024;

/// Standard 1D shader workgroup size for high-throughput elementwise ops.
/// Matches `@workgroup_size(256)` in WGSL shaders.
pub const WORKGROUP_SIZE_1D: u32 = 256;

/// Medium 1D workgroup size for integrated GPUs with lower invocation limits.
/// Used as the `@workgroup_size(128)` tier in elementwise ops (add, mul, fma).
pub const WORKGROUP_SIZE_MEDIUM: u32 = 128;

/// Compact 1D workgroup size for physics/lattice shaders with high register pressure.
/// Matches `@workgroup_size(64)` in MD, lattice QCD, and observable WGSL shaders.
pub const WORKGROUP_SIZE_COMPACT: u32 = 64;

/// Standard 2D shader workgroup size per dimension.
/// Matches `@workgroup_size(16, 16)` in all 2D WGSL shaders.
pub const WORKGROUP_SIZE_2D: u32 = 16;

/// Optimal 1D workgroup size based on GPU architecture.
#[must_use]
pub fn workgroup_size_for_arch(arch: &GpuArch) -> u32 {
    match arch {
        GpuArch::Volta | GpuArch::Turing => 64,
        GpuArch::Ampere | GpuArch::Ada | GpuArch::Blackwell => 256,
        GpuArch::Rdna2 | GpuArch::Rdna3 => 64,
        GpuArch::Cdna2 => 256,
        GpuArch::IntelArc => 128,
        GpuArch::AppleM => 64,
        GpuArch::Software | GpuArch::Unknown => 64,
    }
}

/// 2D workgroup size (per dimension) based on GPU architecture.
#[must_use]
pub fn workgroup_size_2d_for_arch(arch: &GpuArch) -> u32 {
    match arch {
        GpuArch::Ampere | GpuArch::Ada | GpuArch::Blackwell | GpuArch::Cdna2 => 16,
        _ => 8,
    }
}

/// Optimal 1D workgroup size when GPU architecture is known.
#[must_use]
pub fn optimal_workgroup_size_arch(
    arch: &GpuArch,
    workload: WorkloadType,
    max_invocations: u32,
) -> u32 {
    let base = workgroup_size_for_arch(arch);
    let size = match workload {
        WorkloadType::ElementWise | WorkloadType::MatMul | WorkloadType::FHE => base,
        WorkloadType::Reduction => base * 2,
        WorkloadType::Convolution => base / 2,
    };
    size.min(max_invocations)
}

/// NVK (and some other drivers) may report absurd `max_buffer_size` values
/// (e.g. 2^57). Cap to a conservative default when the reported value
/// exceeds a sane maximum.
///
/// # Why name-based heuristics
///
/// wgpu (WebGPU) does **not** expose physical VRAM capacity as a device
/// limit. When the driver reports a bogus value, the only information
/// available at runtime is the adapter name. These heuristics are
/// conservative lower bounds used *only* when the driver is demonstrably
/// broken (>64 GB reported for consumer hardware). When wgpu gains a
/// VRAM capacity limit, this function should switch to it.
fn sanitize_max_buffer_size(reported: u64, device_name: &str) -> u64 {
    const MAX_SANE_BUFFER: u64 = 64 * 1024 * 1024 * 1024;
    const GB: u64 = 1024 * 1024 * 1024;

    const VRAM_HEURISTICS: &[(&[&str], u64)] = &[
        (&["H100", "H200", "B100", "B200", "A100 80GB"], 80 * GB),
        (&["A100", "MI300", "MI250"], 40 * GB),
        (&["RTX 40", "RTX 50"], 24 * GB),
        (&["RTX 30"], 24 * GB),
        (&["Titan V", "V100", "Titan RTX"], 16 * GB),
        (&["RTX 20", "RTX A"], 12 * GB),
    ];
    const VRAM_CAP_CONSERVATIVE: u64 = 8 * GB;

    if reported <= MAX_SANE_BUFFER {
        return reported;
    }

    let capped = VRAM_HEURISTICS
        .iter()
        .find(|(patterns, _)| patterns.iter().any(|p| device_name.contains(p)))
        .map_or(VRAM_CAP_CONSERVATIVE, |&(_, cap)| cap);

    tracing::warn!(
        "Driver reported max_buffer_size={reported} (>64GB), capping to {capped} for {device_name}"
    );
    capped
}

/// Device capabilities - runtime hardware limits
///
/// **Deep Debt**: All values discovered at runtime, zero hardcoding
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name (e.g., "NVIDIA RTX 4090")
    pub device_name: String,

    /// Device type (`DiscreteGpu`, `IntegratedGpu`, Cpu, etc.)
    pub device_type: wgpu::DeviceType,

    /// Maximum buffer size (bytes)
    pub max_buffer_size: u64,

    /// Maximum workgroup size per dimension (x, y, z)
    pub max_workgroup_size: (u32, u32, u32),

    /// Maximum workgroups per dispatch
    pub max_compute_workgroups: (u32, u32, u32),

    /// Maximum invocations per workgroup
    pub max_compute_invocations_per_workgroup: u32,

    /// Maximum storage buffers per shader stage
    pub max_storage_buffers_per_shader_stage: u32,

    /// Maximum uniform buffers per shader stage
    pub max_uniform_buffers_per_shader_stage: u32,

    /// Maximum bind groups
    pub max_bind_groups: u32,

    /// Backend (Vulkan, Metal, DX12, GL, etc.)
    pub backend: wgpu::Backend,

    /// Vendor ID (e.g., `NVIDIA=VENDOR_NVIDIA`, `AMD=VENDOR_AMD`, `Intel=VENDOR_INTEL`)
    pub vendor: u32,

    /// Override for `gpu_dispatch_threshold()`. `None` uses the default per
    /// device type. Set via `with_gpu_dispatch_threshold()`.
    pub gpu_dispatch_threshold_override: Option<usize>,

    /// Minimum subgroup (warp/wavefront) size reported by the adapter.
    /// NVIDIA: 32, AMD RDNA: 32/64, Intel: 8-32, Apple: 32.
    /// Zero if the adapter does not report subgroup info.
    pub subgroup_min_size: u32,

    /// Maximum subgroup (warp/wavefront) size reported by the adapter.
    pub subgroup_max_size: u32,

    /// Whether the device negotiated `wgpu::Features::SUBGROUP`.
    ///
    /// When `true`, subgroup intrinsics (`subgroupAdd`, `subgroupBroadcast`, etc.)
    /// are available in WGSL shaders. Enables subgroup-accelerated reductions
    /// that bypass shared memory for intra-warp communication.
    pub has_subgroups: bool,

    /// Whether the device supports native f64 shader operations.
    pub f64_shaders: bool,

    /// Whether f64 workgroup shared memory (`var<workgroup> x: array<f64, N>`)
    /// produces correct results on this GPU+driver combination.
    ///
    /// groundSpring V84-V85 discovered that f64 shared-memory reductions
    /// return zeros on all current naga/SPIR-V paths (NVIDIA proprietary
    /// and NVK). DF64 shared memory works correctly. This flag gates
    /// whether f64 shared-memory shaders should be attempted or diverted
    /// to DF64.
    pub f64_shared_memory: bool,

    /// Runtime-probed f64 capabilities, or `None` if probing has not occurred.
    ///
    /// When `Some`, capability methods like [`Self::fp64_strategy`] use probed
    /// values for ground-truth behavior. When `None`, they fall back to
    /// heuristics based on device type and features.
    pub f64_capabilities: Option<F64BuiltinCapabilities>,
}

impl DeviceCapabilities {
    /// Detect capabilities from wgpu device
    ///
    /// **Deep Debt**: Runtime discovery, no assumptions
    #[must_use]
    pub fn from_device(device: &WgpuDevice) -> Self {
        let limits = device.device().limits();
        let adapter_info = device.adapter_info();

        Self {
            device_name: adapter_info.name.clone(),
            device_type: adapter_info.device_type,
            max_buffer_size: sanitize_max_buffer_size(limits.max_buffer_size, &adapter_info.name),
            max_workgroup_size: (
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ),
            max_compute_workgroups: (
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ),
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage,
            max_bind_groups: limits.max_bind_groups,
            backend: adapter_info.backend,
            vendor: adapter_info.vendor,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: adapter_info.subgroup_min_size,
            subgroup_max_size: adapter_info.subgroup_max_size,
            has_subgroups: device.has_subgroups(),
            f64_shaders: device.has_f64_shaders(),
            f64_shared_memory: false,
            f64_capabilities: crate::device::probe::cache::cached_f64_builtins(device),
        }
    }

    /// Get optimal workgroup size for a specific workload.
    ///
    /// Selection is based on device type and hardware-reported limits (vendor-agnostic).
    /// The `.min(max_compute_invocations_per_workgroup)` clamp ensures safety on
    /// devices with lower invocation limits.
    #[must_use]
    pub fn optimal_workgroup_size(&self, workload: WorkloadType) -> u32 {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => match workload {
                WorkloadType::ElementWise | WorkloadType::MatMul | WorkloadType::FHE => 256,
                WorkloadType::Reduction => 256,
                WorkloadType::Convolution => 128,
            },

            wgpu::DeviceType::IntegratedGpu => match workload {
                WorkloadType::ElementWise => 128,
                WorkloadType::MatMul => 64,
                WorkloadType::Reduction => 128,
                WorkloadType::FHE => 64,
                WorkloadType::Convolution => 64,
            },

            wgpu::DeviceType::Cpu => match workload {
                WorkloadType::ElementWise => 32,
                WorkloadType::MatMul => 16,
                WorkloadType::Reduction => 64,
                WorkloadType::FHE => 32,
                WorkloadType::Convolution => 16,
            },

            _ => match workload {
                WorkloadType::ElementWise => 64,
                WorkloadType::MatMul => 64,
                WorkloadType::Reduction => 128,
                WorkloadType::FHE => 64,
                WorkloadType::Convolution => 32,
            },
        }
        .min(self.max_compute_invocations_per_workgroup)
    }

    /// Get optimal 2D workgroup size (for 2D operations like convolutions)
    #[must_use]
    pub fn optimal_workgroup_size_2d(&self, workload: WorkloadType) -> (u32, u32) {
        let total = self.optimal_workgroup_size(workload);
        let side = (total as f32).sqrt() as u32;
        let x = side.min(self.max_workgroup_size.0);
        let y = side.min(self.max_workgroup_size.1);
        (x, y)
    }

    /// Get optimal 3D workgroup size (for 3D operations)
    #[must_use]
    pub fn optimal_workgroup_size_3d(&self, workload: WorkloadType) -> (u32, u32, u32) {
        let total = self.optimal_workgroup_size(workload);
        let side = (total as f32).cbrt() as u32;
        let x = side.min(self.max_workgroup_size.0);
        let y = side.min(self.max_workgroup_size.1);
        let z = side.min(self.max_workgroup_size.2);
        (x, y, z)
    }

    /// Calculate number of workgroups for a 1D dispatch.
    #[must_use]
    pub fn dispatch_1d(&self, element_count: u32) -> u32 {
        element_count.div_ceil(WORKGROUP_SIZE_1D)
    }

    /// Calculate number of workgroups for a 2D dispatch.
    #[must_use]
    pub fn dispatch_2d(&self, width: u32, height: u32) -> (u32, u32) {
        (
            width.div_ceil(WORKGROUP_SIZE_2D),
            height.div_ceil(WORKGROUP_SIZE_2D),
        )
    }

    /// Maximum single allocation size (75% of `max_buffer_size`).
    ///
    /// Reserves headroom for driver metadata, command buffers, and
    /// other internal allocations that share the same address space.
    #[must_use]
    pub fn max_allocation_size(&self) -> u64 {
        self.max_buffer_size / 4 * 3
    }

    /// Check if device supports FHE workloads (large U64 buffers)
    #[must_use]
    pub fn supports_fhe(&self) -> bool {
        self.max_buffer_size >= FHE_MIN_BUFFER_SIZE
    }

    /// Check if device supports large matrix operations
    #[must_use]
    pub fn supports_large_matmul(&self, m: usize, n: usize, k: usize) -> bool {
        let required_bytes = (m * k + k * n + m * n) * 4;
        required_bytes as u64 <= self.max_allocation_size()
    }

    /// Get optimal tile size for matrix multiplication.
    ///
    /// Discrete GPUs with high invocation limits use 32×32 tiles; devices with
    /// lower compute throughput use smaller tiles. Selection is based on device
    /// type and hardware limits, not vendor identity.
    #[must_use]
    pub fn optimal_matmul_tile_size(&self) -> u32 {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => {
                if self.max_compute_invocations_per_workgroup >= 1024 {
                    32
                } else {
                    16
                }
            }
            wgpu::DeviceType::IntegratedGpu => 16,
            wgpu::DeviceType::Cpu => 8,
            _ => 8,
        }
    }

    /// Minimum element count below which CPU is faster than a GPU dispatch.
    ///
    /// Conservative defaults by device class — override with
    /// [`Self::with_gpu_dispatch_threshold`] for workload-specific tuning.
    #[must_use]
    pub fn gpu_dispatch_threshold(&self) -> usize {
        const DISCRETE_THRESHOLD: usize = 4_096;
        const INTEGRATED_THRESHOLD: usize = 16_384;
        const OTHER_THRESHOLD: usize = 8_192;

        if let Some(t) = self.gpu_dispatch_threshold_override {
            return t;
        }
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => DISCRETE_THRESHOLD,
            wgpu::DeviceType::IntegratedGpu => INTEGRATED_THRESHOLD,
            wgpu::DeviceType::Cpu => usize::MAX,
            _ => OTHER_THRESHOLD,
        }
    }

    /// Return a copy with the GPU dispatch threshold set to `threshold`.
    #[must_use]
    pub fn with_gpu_dispatch_threshold(mut self, threshold: usize) -> Self {
        self.gpu_dispatch_threshold_override = Some(threshold);
        self
    }

    /// Get vendor name (for logging/debugging)
    #[must_use]
    pub fn vendor_name(&self) -> &'static str {
        match self.vendor {
            VENDOR_NVIDIA => "NVIDIA",
            VENDOR_AMD => "AMD",
            VENDOR_INTEL => "Intel",
            0x13B5 => "ARM",
            0x5143 => "Qualcomm",
            0x1010 => "ImgTec",
            _ => "Unknown",
        }
    }

    /// Check if this is a high-performance GPU
    #[must_use]
    pub fn is_high_performance(&self) -> bool {
        matches!(self.device_type, wgpu::DeviceType::DiscreteGpu)
            && self.max_compute_invocations_per_workgroup >= HIGH_PERFORMANCE_MIN_INVOCATIONS
    }

    /// Whether the adapter reports subgroup (warp/wavefront) sizes.
    ///
    /// True when `subgroup_min_size > 0`, indicating the driver exposes
    /// subgroup metadata. For actual subgroup intrinsic support (e.g.
    /// `subgroupAdd`), check [`has_subgroups`](Self::has_subgroups) instead.
    #[must_use]
    pub fn has_subgroup_info(&self) -> bool {
        self.subgroup_min_size > 0
    }

    /// Preferred subgroup size for reduction shaders.
    ///
    /// Returns the max subgroup size (to maximise lane utilisation in
    /// tree reductions), or `None` when not reported.
    #[must_use]
    pub fn preferred_subgroup_size(&self) -> Option<u32> {
        if self.subgroup_max_size > 0 {
            Some(self.subgroup_max_size)
        } else {
            None
        }
    }

    /// Whether f64 workgroup shared memory produces correct results.
    ///
    /// Currently `false` for all naga/SPIR-V paths due to a systemic bug
    /// in naga's SPIR-V emission for f64 workgroup shared memory. Shaders
    /// using `var<workgroup> x: array<f64, N>` should divert to DF64 when
    /// this returns `false`.
    #[must_use]
    pub fn has_f64_shared_memory(&self) -> bool {
        self.f64_shared_memory
    }

    /// Enrich with runtime-probed f64 capabilities.
    ///
    /// Call after [`crate::device::probe::probe_f64_builtins`] to populate
    /// probe-based capability queries (`fp64_strategy`, `needs_*_workaround`, etc.).
    #[must_use]
    pub fn with_f64_capabilities(mut self, caps: F64BuiltinCapabilities) -> Self {
        self.f64_capabilities = Some(caps);
        self
    }

    // ── Probe-based capability queries ──────────────────────────────────

    /// Whether f64 compute shaders produce correct results on this device.
    ///
    /// Probe-based when available; falls back to the `f64_shaders` feature flag.
    #[must_use]
    pub fn has_reliable_f64(&self) -> bool {
        self.f64_capabilities
            .map_or(self.f64_shaders, |c| c.basic_f64)
    }

    /// Whether `exp(f64)` needs a software workaround on this device.
    #[must_use]
    pub fn needs_exp_f64_workaround(&self) -> bool {
        self.f64_capabilities.is_some_and(|c| !c.exp)
    }

    /// Whether `log(f64)` needs a software workaround on this device.
    #[must_use]
    pub fn needs_log_f64_workaround(&self) -> bool {
        self.f64_capabilities.is_some_and(|c| !c.log)
    }

    /// Whether `sin(f64)` needs a software workaround on this device.
    #[must_use]
    pub fn needs_sin_f64_workaround(&self) -> bool {
        self.f64_capabilities.is_some_and(|c| !c.sin)
    }

    /// Whether `cos(f64)` needs a software workaround on this device.
    #[must_use]
    pub fn needs_cos_f64_workaround(&self) -> bool {
        self.f64_capabilities.is_some_and(|c| !c.cos)
    }

    /// Whether `sqrt(f64)` needs a software workaround on this device.
    #[must_use]
    pub fn needs_sqrt_f64_workaround(&self) -> bool {
        self.f64_capabilities.is_some_and(|c| !c.sqrt)
    }

    /// Whether all f64 transcendentals (sqrt, sin, cos, exp, log, abs, fma)
    /// work correctly on this device with full f64 precision.
    ///
    /// When `false`, shaders using transcendentals should route through
    /// polyfill, DF64, or CPU fallback. Probe-based when available.
    #[must_use]
    pub fn has_f64_transcendentals(&self) -> bool {
        self.f64_capabilities
            .map_or(self.f64_shaders, |c| c.has_f64_transcendentals())
    }

    /// Whether `DF64` transcendentals are safe on this device.
    #[must_use]
    pub fn df64_transcendentals_safe(&self) -> bool {
        self.f64_capabilities
            .is_none_or(|c| c.df64_transcendentals_safe)
    }

    /// Hardware-adaptive FP64 execution strategy (capability-based).
    ///
    /// Uses probed f64 capabilities when available; falls back to heuristics
    /// based on device type and `f64_shaders` feature flag.
    #[must_use]
    pub fn fp64_strategy(&self) -> super::Fp64Strategy {
        use super::Fp64Strategy;

        if let Some(caps) = &self.f64_capabilities {
            if !caps.basic_f64 {
                return Fp64Strategy::Hybrid;
            }
            if caps.can_compile_f64() {
                return Fp64Strategy::Native;
            }
            return Fp64Strategy::Hybrid;
        }

        if self.f64_shaders {
            Fp64Strategy::Native
        } else {
            Fp64Strategy::Hybrid
        }
    }

    /// Precision routing advice (capability-based).
    ///
    /// Higher-level than `fp64_strategy()`: additionally captures the
    /// shared-memory reliability axis for workgroup-based f64 reductions.
    #[must_use]
    pub fn precision_routing(&self) -> super::super::driver_profile::PrecisionRoutingAdvice {
        use super::super::driver_profile::PrecisionRoutingAdvice;

        if let Some(caps) = &self.f64_capabilities {
            if !caps.basic_f64 {
                if self.f64_shaders {
                    return PrecisionRoutingAdvice::Df64Only;
                }
                return PrecisionRoutingAdvice::F32Only;
            }
            if caps.shared_mem_f64 {
                return PrecisionRoutingAdvice::F64Native;
            }
            return PrecisionRoutingAdvice::F64NativeNoSharedMem;
        }

        if self.f64_shaders {
            PrecisionRoutingAdvice::F64NativeNoSharedMem
        } else {
            PrecisionRoutingAdvice::F32Only
        }
    }

    /// Eigensolve workgroup size based on subgroup (warp/wavefront) size.
    ///
    /// Vendor-agnostic: uses the reported subgroup size to determine the
    /// natural SIMT width for warp/wave-packed eigensolve dispatch.
    /// Falls back to 32 (the most common subgroup size across vendors).
    #[must_use]
    pub fn eigensolve_workgroup_size(&self) -> u32 {
        if self.subgroup_min_size > 0 {
            self.subgroup_min_size
        } else {
            32
        }
    }

    /// Optimal eigensolve dispatch strategy based on device capabilities.
    ///
    /// Uses subgroup size and device type to determine packing:
    /// - Discrete GPUs with known subgroup size → warp/wave-packed
    /// - CPU/Software/unknown → standard (no packing)
    #[must_use]
    pub fn optimal_eigensolve_strategy(&self) -> super::EigensolveStrategy {
        use super::EigensolveStrategy;
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu => {
                let wg = self.eigensolve_workgroup_size();
                if wg >= 64 {
                    EigensolveStrategy::WavePacked { wave_size: wg }
                } else {
                    EigensolveStrategy::WarpPacked { wg_size: wg }
                }
            }
            _ => EigensolveStrategy::Standard,
        }
    }

    /// Maximum safe combined allocation in bytes, or `None` if no known limit.
    ///
    /// Conservative estimate: 75% of `max_buffer_size` to leave headroom for
    /// driver metadata and command buffers. Individual drivers may have tighter
    /// limits discovered via runtime probing.
    #[must_use]
    pub fn max_safe_allocation_bytes(&self) -> Option<u64> {
        Some(self.max_allocation_size())
    }

    /// Whether native f64 builtins (exp, log, pow) work on this device.
    ///
    /// Probe-based when available; falls back to the `f64_shaders` flag and
    /// assumes builtins work unless probing says otherwise.
    #[must_use]
    pub fn supports_f64_builtins(&self) -> bool {
        if let Some(caps) = &self.f64_capabilities {
            caps.can_compile_f64() && caps.exp && caps.log
        } else {
            self.f64_shaders
        }
    }

    /// Whether DF64 shaders containing transcendentals produce poisoned output.
    ///
    /// Inverse of [`df64_transcendentals_safe`](Self::df64_transcendentals_safe).
    #[must_use]
    pub fn has_df64_spir_v_poisoning(&self) -> bool {
        !self.df64_transcendentals_safe()
    }

    /// Return a latency model for ILP scheduling based on device characteristics.
    ///
    /// Uses vendor ID and device type to select the closest empirical model:
    /// - NVIDIA discrete GPUs → SM70 model (8-cycle DFMA pipeline)
    /// - AMD discrete GPUs → RDNA2 model (4-cycle VFMA64 pipeline)
    /// - Apple GPUs → Apple M model (software-emulated f64)
    /// - All others → conservative model (safe over-estimate)
    ///
    /// When runtime-measured latency data is available (via `bench_f64_builtins`),
    /// callers should prefer `MeasuredModel` directly.
    #[must_use]
    pub fn latency_model(&self) -> Box<dyn crate::device::latency::LatencyModel> {
        use crate::device::latency::{
            AppleMLatencyModel, ConservativeModel, Rdna2LatencyModel, Sm70LatencyModel,
        };
        use crate::device::vendor::{VENDOR_AMD, VENDOR_APPLE, VENDOR_NVIDIA};

        match self.device_type {
            wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu => match self.vendor {
                VENDOR_NVIDIA => Box::new(Sm70LatencyModel),
                VENDOR_AMD => Box::new(Rdna2LatencyModel),
                VENDOR_APPLE => Box::new(AppleMLatencyModel),
                _ => Box::new(ConservativeModel),
            },
            _ => Box::new(ConservativeModel),
        }
    }

    /// Check whether a combined allocation of `total_bytes` is safe on this device.
    ///
    /// # Errors
    ///
    /// Returns [`DeviceLimitExceeded`](crate::error::BarracudaError::DeviceLimitExceeded)
    /// if the allocation exceeds the safe limit.
    pub fn check_allocation_safe(&self, total_bytes: u64) -> crate::error::Result<()> {
        if let Some(limit) = self.max_safe_allocation_bytes() {
            if total_bytes > limit {
                return Err(crate::error::BarracudaError::DeviceLimitExceeded {
                    message: format!(
                        "Estimated allocation {:.1} MB exceeds safe limit {:.1} MB",
                        total_bytes as f64 / 1e6,
                        limit as f64 / 1e6,
                    ),
                    requested_bytes: total_bytes,
                    safe_limit_bytes: limit,
                });
            }
        }
        Ok(())
    }
}

/// Workload types for optimal configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// Element-wise ops (activation, etc.).
    ElementWise,
    /// Matrix multiplication.
    MatMul,
    /// Reduction ops (sum, max, etc.).
    Reduction,
    /// Fully homomorphic encryption workloads.
    FHE,
    /// Convolution operations.
    Convolution,
}

mod display;
