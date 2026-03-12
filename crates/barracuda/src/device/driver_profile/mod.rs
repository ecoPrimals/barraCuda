// SPDX-License-Identifier: AGPL-3.0-only
//! GPU Driver Profile — data-driven shader specialisation.
//!
//! This module answers the question **"who is driving the hardware?"** and
//! translates that into concrete shader strategies.  It complements
//! `capabilities` (which answers "what can the hardware do?") by providing
//! the compiler/driver layer between the application and the silicon.
//!
//! ## Design
//!
//! A single `GpuDriverProfile` struct consolidates:
//! - Driver identity (`DriverKind`: NVK, RADV, proprietary NVIDIA, …)
//! - Shader compiler backend (`CompilerKind`: NAK, ACO, PTXAS, …)
//! - GPU microarchitecture (`GpuArch`: Volta, Turing, RDNA2, …)
//! - FP64 hardware rate classification (`Fp64Rate`)
//! - Active workarounds (`Workaround`: NVK exp/log crash, …)
//!
//! Query `GpuDriverProfile` at dispatch time instead of re-running
//! string-matching logic scattered across the codebase.
//!
//! ## Sovereign Compute Evolution
//!
//! All four Sovereign phases are tracked here:
//! - Phase 1 ✓  Profile detection + eigensolve strategy
//! - Phase 2 ✓  NAK contribution (SM70/RDNA2/AppleM latency tables)
//! - Phase 3 ✓  ILP reorderer + loop unroller wired into `compile_shader_f64()`
//! - Phase 4    Specialised codegen — deferred until upstream bottleneck confirmed
//!
//! Reference: phases described above; originated from orchestration layer sovereign compute spec.

mod architectures;
mod workarounds;

pub use architectures::GpuArch;
pub use workarounds::Workaround;

use std::fmt;

use crate::device::WgpuDevice;
use crate::device::probe::cache::adapter_key;
use crate::error::{BarracudaError, Result};

// ── Driver identity ───────────────────────────────────────────────────────────

/// GPU driver/compiler identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriverKind {
    /// NVIDIA proprietary (NVVM/PTXAS)
    NvidiaProprietary,
    /// Mesa NVK (open-source NVIDIA Vulkan)
    Nvk,
    /// Mesa RADV (AMD Vulkan)
    Radv,
    /// Intel ANV
    Intel,
    /// Software rasterizer
    Software,
    /// Unknown driver
    Unknown,
}

/// GPU shader compiler backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompilerKind {
    /// NVIDIA proprietary PTX assembler
    NvidiaPtxas,
    /// Mesa NAK (Rust-based NVIDIA compiler)
    Nak,
    /// Mesa ACO (AMD compiler)
    Aco,
    /// Intel ANV compiler
    Anv,
    /// Software rasterizer (llvmpipe, swiftshader)
    Software,
    /// Unknown compiler
    Unknown,
}

// ── FP64 rate ─────────────────────────────────────────────────────────────────

/// FP64 hardware rate classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp64Rate {
    /// Full rate: FP64:FP32 = 1:2 (Titan V, MI250, etc.)
    Full,
    /// Throttled by vendor SDK but accessible via Vulkan
    Throttled,
    /// Hardware rate 1:64 (consumer Ada, Turing)
    Minimal,
    /// Software emulated
    Software,
}

// ── Eigensolve strategy ───────────────────────────────────────────────────────

/// Eigensolve dispatch strategy chosen based on driver/arch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EigensolveStrategy {
    /// Warp-packed: 32 independent matrices per workgroup (NVIDIA)
    WarpPacked {
        /// Workgroup size (typically 32)
        wg_size: u32,
    },
    /// Wave-packed: 64 independent matrices per workgroup (AMD)
    WavePacked {
        /// Wavefront size (typically 64)
        wave_size: u32,
    },
    /// Standard: one matrix per workgroup
    Standard,
}

// ── FP64 execution strategy ──────────────────────────────────────────────────

/// Hardware-adaptive FP64 execution strategy (physics domain core-streaming discovery).
///
/// On compute-class GPUs (Titan V, V100, MI250) with 1:2 FP64:FP32 hardware,
/// native `f64` is fastest. On consumer GPUs (1:64 ratio), routing bulk math
/// through double-float f32-pair (`Df64`) on the massive FP32 core array
/// delivers ~10x the effective throughput, reserving native `f64` for
/// precision-critical reductions and convergence tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp64Strategy {
    /// Sovereign compilation via coralReef: WGSL → native GPU binary via IPC.
    /// Bypasses naga/SPIR-V entirely — resolves f64 shared-memory failures.
    /// Requires coralReef primal + coralDriver for dispatch; falls back
    /// through the chain: Sovereign → Native → Hybrid → f32.
    Sovereign,
    /// Use native f64 everywhere (Titan V, V100, A100, MI250X — 1:2 hardware).
    Native,
    /// DF64 (f32-pair, ~14 digits) for bulk math, native f64 for reductions.
    Hybrid,
    /// Run both DF64 and native f64 concurrently, cross-validate results.
    /// Useful for validation harnesses and precision-sensitive pipelines.
    Concurrent,
}

// ── Precision routing (toadStool S128 integration) ──────────────────────────

/// Precision routing advice from toadStool S128.
///
/// Captures the three-axis reality of GPU f64: some hardware has native f64
/// everywhere, some has native f64 but broken shared-memory f64 reductions
/// (naga/SPIR-V emit zeros), some can only do DF64, and some are f32-only.
///
/// Use this to route workloads to the correct precision path at dispatch time,
/// especially for reductions that rely on `var<workgroup>` f64 accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionRoutingAdvice {
    /// Native f64 everywhere — compute and shared-memory reductions.
    F64Native,
    /// Native f64 compute works, but shared-memory f64 reductions return zeros.
    /// Route reductions through DF64 or scalar f64 (no workgroup accumulator).
    F64NativeNoSharedMem,
    /// No reliable native f64; use DF64 (f32-pair) for all f64-class work.
    Df64Only,
    /// No f64 support at all; fall back to f32.
    F32Only,
}

// ── GpuDriverProfile ──────────────────────────────────────────────────────────

/// Unified GPU driver profile for data-driven shader specialisation.
///
/// Consolidates driver detection, compiler quality knowledge, and known
/// workarounds. Query this instead of string-matching device names at
/// dispatch time.
///
/// ## Construction
///
/// ```rust,no_run
/// # use barracuda::device::{WgpuDevice, driver_profile::GpuDriverProfile};
/// # async fn example() -> barracuda::error::Result<()> {
/// let device = WgpuDevice::new().await?;
/// let profile = GpuDriverProfile::from_device(&device);
/// println!("{profile}");
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct GpuDriverProfile {
    /// Driver identity
    pub driver: DriverKind,
    /// Shader compiler backend
    pub compiler: CompilerKind,
    /// GPU microarchitecture
    pub arch: GpuArch,
    /// FP64 hardware rate classification
    pub fp64_rate: Fp64Rate,
    /// Active workarounds for this driver/arch
    pub workarounds: Vec<Workaround>,
    /// Adapter cache key for probe result lookup
    pub(crate) adapter_key: String,
}

impl GpuDriverProfile {
    /// Build a driver profile from a `WgpuDevice` using runtime detection.
    #[must_use]
    pub fn from_device(device: &WgpuDevice) -> Self {
        let driver = Self::detect_driver(device);
        let compiler = Self::detect_compiler(driver);
        let arch = architectures::detect_arch(device);
        let fp64_rate = Self::detect_fp64_rate(arch, driver);
        let workarounds = workarounds::detect_workarounds(driver, arch);
        let adapter_key = adapter_key(device);

        Self {
            driver,
            compiler,
            arch,
            fp64_rate,
            workarounds,
            adapter_key,
        }
    }

    /// Optimal eigensolve dispatch strategy for this driver/arch combination.
    /// Physics domain measured 2.2× NVK speedup with warp-packing (Titan V, Feb 2026).
    /// Neutral on proprietary NVIDIA (scheduler already handles wg1 efficiently).
    /// ### AMD RDNA2/RDNA3 (ACO compiler)
    /// Empirically measured on RX 6950 XT (RDNA2/NAVI21, Feb 2026):
    /// - `wg_size=32`: 67.7 ms  ← optimal
    /// - `wg_size=64`: 117.1 ms ← 1.7× slower
    ///   Root cause: ACO targets **wave32 mode** for compute shaders on RDNA2.
    ///   Use `WarpPacked { wg_size: 32 }` for all current ACO targets.
    #[must_use]
    pub fn optimal_eigensolve_strategy(&self) -> EigensolveStrategy {
        match (self.compiler, self.arch) {
            (CompilerKind::Nak, _) => EigensolveStrategy::WarpPacked { wg_size: 32 },
            (CompilerKind::Aco, GpuArch::Rdna2 | GpuArch::Rdna3) => {
                EigensolveStrategy::WarpPacked { wg_size: 32 }
            }
            (CompilerKind::Aco, GpuArch::Cdna2) => EigensolveStrategy::WavePacked { wave_size: 64 },
            (CompilerKind::NvidiaPtxas, _) => EigensolveStrategy::WarpPacked { wg_size: 32 },
            _ => EigensolveStrategy::Standard,
        }
    }

    /// Whether `exp(f64)` needs software substitution on this driver.
    #[must_use]
    pub fn needs_exp_f64_workaround(&self) -> bool {
        self.workarounds.contains(&Workaround::NvkExpF64Crash)
            || self
                .workarounds
                .contains(&Workaround::NvvmAdaF64Transcendentals)
    }

    /// Whether `log(f64)` needs software substitution on this driver.
    #[must_use]
    pub fn needs_log_f64_workaround(&self) -> bool {
        self.workarounds.contains(&Workaround::NvkLogF64Crash)
            || self
                .workarounds
                .contains(&Workaround::NvvmAdaF64Transcendentals)
    }

    /// Whether `pow(f64, f64)` needs software substitution on this driver.
    /// Ada Lovelace (SM89) with the proprietary NVIDIA driver cannot compile
    /// native f64 pow/exp/log. Discovered by marine biology domain on RTX 4070 (Feb 2026).
    #[must_use]
    pub fn needs_pow_f64_workaround(&self) -> bool {
        self.workarounds
            .contains(&Workaround::NvvmAdaF64Transcendentals)
    }

    /// Whether `sin(f64)` needs Taylor-series workaround on this driver.
    /// NVK (open-source NVIDIA Vulkan driver) has broken or imprecise native
    /// sin/cos implementations for f64. Returns true for NVK and any driver
    /// with the `NvkSinCosF64Imprecise` workaround.
    #[must_use]
    pub fn needs_sin_f64_workaround(&self) -> bool {
        self.workarounds
            .contains(&Workaround::NvkSinCosF64Imprecise)
    }

    /// Whether `cos(f64)` needs Taylor-series workaround on this driver.
    /// NVK has broken or imprecise native cos for f64. Returns true for NVK
    /// and any driver with the `NvkSinCosF64Imprecise` workaround.
    #[must_use]
    pub fn needs_cos_f64_workaround(&self) -> bool {
        self.workarounds
            .contains(&Workaround::NvkSinCosF64Imprecise)
    }

    /// Whether this driver supports f64 builtins (exp, log, pow) natively.
    /// Returns false for NVK, software renderers, and NVIDIA Ada Lovelace
    /// proprietary (NVVM PTXAS fails on f64 transcendentals for SM89).
    #[must_use]
    pub fn supports_f64_builtins(&self) -> bool {
        self.workarounds.is_empty() && !matches!(self.driver, DriverKind::Software)
    }

    /// Whether this device requires nvPmu (software PMU) initialization
    /// before sovereign compute dispatch.
    ///
    /// True for Volta desktop GPUs (Titan V, GV100) on NVK — NVIDIA does
    /// not ship PMU firmware for desktop Volta, so nouveau cannot create
    /// compute channels without software-assisted engine initialization.
    #[must_use]
    pub fn needs_software_pmu(&self) -> bool {
        self.workarounds.contains(&Workaround::VoltaNoPmuFirmware)
    }

    /// Whether sovereign dispatch (coralReef + coral-driver) can bypass
    /// naga/SPIR-V poisoning on this device.
    ///
    /// Returns `true` for devices where native binary dispatch resolves
    /// the DF64 transcendental poisoning — i.e., any device with
    /// `Df64SpirVPoisoning` that has a sovereign compilation path available.
    /// Devices requiring nvPmu init should check `needs_software_pmu()` first.
    #[must_use]
    pub fn sovereign_resolves_poisoning(&self) -> bool {
        self.has_df64_spir_v_poisoning()
    }

    /// Whether DF64 shaders containing transcendentals (exp, sqrt, log, pow)
    /// produce all-zero output due to naga WGSL->SPIR-V codegen bugs.
    ///
    /// Affects ALL Vulkan backends (NVIDIA proprietary, NVK/NAK, llvmpipe).
    /// Root cause is naga's SPIR-V generation for mixed f64/f32 shaders, not
    /// the driver JIT. On NVIDIA proprietary, the bad SPIR-V additionally
    /// causes unrecoverable device invalidation.
    ///
    /// When `true`, callers must either strip transcendentals from the DF64
    /// preamble, avoid DF64 compilation, or use sovereign dispatch (coralReef)
    /// which bypasses naga SPIR-V entirely.
    ///
    /// Discovered by hotSpring v0.6.25 (March 2026). Root-caused by hotSpring
    /// Exp 055 (March 2026). Sovereign fix validated by coralReef Iter 33.
    #[must_use]
    pub fn has_df64_spir_v_poisoning(&self) -> bool {
        self.workarounds.contains(&Workaround::Df64SpirVPoisoning)
    }

    /// Whether this driver reliably supports f64 shader operations.
    ///
    /// Returns `true` only when the runtime probe confirms f64 compilation
    /// produces correct results (`f64(3)*f64(2)+f64(1)==7.0`). NVK (Titan V,
    /// RTX 4070) and NVIDIA proprietary on Ada advertise `SHADER_F64` but fail
    /// to compile or produce correct results. toadStool S97, hotSpring, and
    /// groundSpring all require probe-verified f64 before dispatching native
    /// f64 shaders.
    ///
    /// When no probe result is cached, falls back to heuristic detection
    /// (workaround-free + not a software renderer).
    #[must_use]
    pub fn has_reliable_f64(&self) -> bool {
        match crate::device::probe::cache::cached_basic_f64_for_key(&self.adapter_key) {
            Some(probed) => probed,
            None => self.workarounds.is_empty() && !matches!(self.driver, DriverKind::Software),
        }
    }

    /// Precision routing advice integrating toadStool S128 f64 shared-memory discovery.
    ///
    /// This is higher-level than `fp64_strategy()`: it additionally captures
    /// the shared-memory reliability axis. Use this to decide whether a
    /// workgroup-based f64 reduction can use `var<workgroup>` accumulators
    /// or must fall back to scalar/DF64 paths.
    ///
    /// The hierarchy:
    /// - `F64Native` — full native f64 everywhere (Titan V with working driver)
    /// - `F64NativeNoSharedMem` — native f64 compute works, but shared-memory
    ///   reductions return zeros (common on NVK/NAK today)
    /// - `Df64Only` — use DF64 for all f64-class work
    /// - `F32Only` — no f64 support at all
    #[must_use]
    pub fn precision_routing(&self) -> PrecisionRoutingAdvice {
        let native_fails = matches!(
            crate::device::probe::cache::cached_basic_f64_for_key(&self.adapter_key),
            Some(false)
        );

        if native_fails {
            if matches!(self.fp64_rate, Fp64Rate::Software) {
                return PrecisionRoutingAdvice::F32Only;
            }
            return PrecisionRoutingAdvice::Df64Only;
        }

        let shared_mem_fails = matches!(
            crate::device::probe::cache::cached_shared_mem_f64_for_key(&self.adapter_key),
            Some(false)
        );

        match self.fp64_rate {
            Fp64Rate::Full => {
                if shared_mem_fails || self.driver == DriverKind::Nvk {
                    PrecisionRoutingAdvice::F64NativeNoSharedMem
                } else {
                    PrecisionRoutingAdvice::F64Native
                }
            }
            Fp64Rate::Throttled | Fp64Rate::Minimal => {
                if shared_mem_fails
                    || (self.driver == DriverKind::NvidiaProprietary
                        && self.arch == architectures::GpuArch::Ada)
                {
                    PrecisionRoutingAdvice::F64NativeNoSharedMem
                } else {
                    PrecisionRoutingAdvice::Df64Only
                }
            }
            Fp64Rate::Software => PrecisionRoutingAdvice::F32Only,
        }
    }

    /// Choose the optimal FP64 execution strategy for this hardware.
    ///
    /// Probe-aware: if the runtime probe has determined that native f64
    /// compilation fails (NAK, NVVM returning zeros), forces Hybrid even on
    /// hardware with Full FP64 rate.  Falls back to the hardware-rate
    /// heuristic when no probe result is cached yet.
    ///
    /// When coralReef is discovered at runtime (via `probe_health()`), and
    /// native f64 probes fail, returns `Sovereign` — indicating that
    /// coralReef-compiled native binaries should be used once coralDriver
    /// is available for dispatch.
    #[must_use]
    pub fn fp64_strategy(&self) -> Fp64Strategy {
        let native_f64_fails = matches!(
            crate::device::probe::cache::cached_basic_f64_for_key(&self.adapter_key),
            Some(false)
        );

        if native_f64_fails {
            return Fp64Strategy::Hybrid;
        }
        match self.fp64_rate {
            Fp64Rate::Full => Fp64Strategy::Native,
            Fp64Rate::Throttled | Fp64Rate::Minimal | Fp64Rate::Software => Fp64Strategy::Hybrid,
        }
    }

    /// Probe-informed FP64 strategy. Overrides heuristic when the runtime
    /// probe shows f64 compilation actually fails (NAK, NVVM).
    /// Earth science domain V35/V37 discovery: NAK and NVVM advertise `SHADER_F64`
    /// but cannot compile f64 WGSL. The probe provides ground truth.
    #[must_use]
    pub fn fp64_strategy_probed(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> Fp64Strategy {
        if !caps.can_compile_f64() {
            return Fp64Strategy::Hybrid;
        }
        self.fp64_strategy()
    }

    /// Whether a shader containing f64 transcendentals can safely use the
    /// Hybrid (DF64) strategy on this device.
    ///
    /// On NVIDIA proprietary, DF64 transcendentals cause unrecoverable NVVM
    /// device poisoning. When `false`, callers must either use native F64
    /// (if available) or fall back to F32 for shaders with transcendentals.
    #[must_use]
    pub fn df64_transcendentals_safe(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> bool {
        caps.df64_transcendentals_safe
    }

    /// Whether `sin(f64)` needs software substitution, considering probe results.
    #[must_use]
    pub fn needs_sin_f64_workaround_probed(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> bool {
        caps.needs_sin_f64_workaround()
    }

    /// Whether `cos(f64)` needs software substitution, considering probe results.
    #[must_use]
    pub fn needs_cos_f64_workaround_probed(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> bool {
        caps.needs_cos_f64_workaround()
    }

    /// Whether this is an open-source driver (NVK or RADV).
    #[must_use]
    pub fn is_open_source(&self) -> bool {
        matches!(self.driver, DriverKind::Nvk | DriverKind::Radv)
    }

    /// Whether this device may advertise f64 support but produce zero outputs.
    ///
    /// NVK on Titan V and some consumer NVIDIA GPUs advertise `Float64` in
    /// wgpu features, but shared-memory f64 accumulators return zeros for
    /// reduction operations. Ada Lovelace on proprietary drivers exhibits
    /// the same shared-memory f64 reduction failures despite basic f64 compute
    /// working correctly. This flag allows springs to skip or guard f64
    /// reduction tests on affected hardware.
    #[must_use]
    pub fn f64_zeros_risk(&self) -> bool {
        let nvk_risk = self.driver == DriverKind::Nvk
            && matches!(self.fp64_rate, Fp64Rate::Full | Fp64Rate::Throttled);
        let ada_proprietary_risk = self.driver == DriverKind::NvidiaProprietary
            && self.arch == architectures::GpuArch::Ada;
        nvk_risk || ada_proprietary_risk
    }

    /// Return the `LatencyModel` appropriate for this GPU architecture.
    /// The model provides per-operation cycle counts used by the WGSL ILP
    /// scheduler (`@ilp_region` reorderer, Phase 3 `WgslDependencyGraph`).
    /// - NVIDIA Volta/Turing/Ampere/Ada → `Sm70LatencyModel` (DFMA = 8 cy)
    /// - AMD RDNA2/RDNA3/CDNA2 → `Rdna2LatencyModel` (VFMA64 ≈ 4 cy)
    /// - Unknown/Intel/Software → `ConservativeModel` (safe overestimate)
    #[must_use]
    pub fn latency_model(&self) -> Box<dyn crate::device::latency::LatencyModel> {
        crate::device::latency::model_for_arch(self.arch)
    }

    // ── Allocation safety ─────────────────────────────────────────────────────

    /// Maximum safe combined allocation in bytes, or `None` if unlimited.
    /// On NVK (nouveau), the kernel driver can PTE-fault when combined GPU
    /// allocations exceed ~1.4 GB (observed on GV100/Titan V). We use a
    /// conservative 1.2 GB limit to avoid silent device loss.
    #[must_use]
    pub fn max_safe_total_allocation(&self) -> Option<u64> {
        if self.workarounds.contains(&Workaround::NvkLargeBufferLimit) {
            Some(workarounds::NVK_MAX_SAFE_ALLOCATION_BYTES)
        } else {
            None
        }
    }

    /// Check whether a combined allocation of `total_bytes` is safe on this
    /// driver.
    /// # Errors
    /// Returns [`Err`] with [`DeviceLimitExceeded`] if the allocation exceeds
    /// the driver's safe limit (e.g. NVK large-buffer limit); suggests Mesa
    /// git HEAD in the diagnostic.
    pub fn check_allocation_safe(&self, total_bytes: u64) -> Result<()> {
        if let Some(limit) = self.max_safe_total_allocation() {
            if total_bytes > limit {
                tracing::warn!(
                    total_bytes,
                    safe_limit = limit,
                    "NVK large-buffer limit exceeded — nouveau PTE fault likely. \
                     Consider using Mesa git HEAD which may have the drm_gpuvm fix."
                );
                return Err(BarracudaError::DeviceLimitExceeded {
                    message: format!(
                        "NVK (nouveau) driver unsafe for {:.1} MB combined allocation; \
                         limit is {:.1} MB to avoid kernel PTE fault. \
                         Try Mesa git HEAD or the proprietary NVIDIA driver.",
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

    // ── Internal detection helpers ────────────────────────────────────────────

    fn detect_driver(device: &WgpuDevice) -> DriverKind {
        if device.is_nvk() {
            DriverKind::Nvk
        } else if device.is_nvidia_proprietary() {
            DriverKind::NvidiaProprietary
        } else if device.is_radv() {
            DriverKind::Radv
        } else {
            let name = device.adapter_info().name.to_lowercase();
            if name.contains("intel") || name.contains("iris") {
                DriverKind::Intel
            } else if name.contains("llvmpipe")
                || name.contains("swiftshader")
                || name.contains("software")
            {
                DriverKind::Software
            } else {
                DriverKind::Unknown
            }
        }
    }

    fn detect_compiler(driver: DriverKind) -> CompilerKind {
        match driver {
            DriverKind::NvidiaProprietary => CompilerKind::NvidiaPtxas,
            DriverKind::Nvk => CompilerKind::Nak,
            DriverKind::Radv => CompilerKind::Aco,
            DriverKind::Intel => CompilerKind::Anv,
            DriverKind::Software => CompilerKind::Software,
            DriverKind::Unknown => CompilerKind::Unknown,
        }
    }

    fn detect_fp64_rate(arch: GpuArch, driver: DriverKind) -> Fp64Rate {
        match arch {
            GpuArch::Volta => Fp64Rate::Full,
            GpuArch::Ampere => Fp64Rate::Throttled,
            GpuArch::Ada => Fp64Rate::Throttled,
            GpuArch::Turing => Fp64Rate::Throttled,
            GpuArch::Rdna2 | GpuArch::Rdna3 => Fp64Rate::Throttled,
            GpuArch::Cdna2 => Fp64Rate::Full,
            GpuArch::IntelArc => Fp64Rate::Minimal,
            GpuArch::AppleM => Fp64Rate::Software,
            GpuArch::Software => Fp64Rate::Software,
            GpuArch::Unknown => {
                if matches!(driver, DriverKind::Software) {
                    Fp64Rate::Software
                } else {
                    Fp64Rate::Throttled
                }
            }
        }
    }

    /// Preferred 1D workgroup size for the GPU architecture.
    /// Volta/Turing (SM70/SM75): 64-wide warps → 64.
    /// Ampere/Ada (SM80+) and AMD RDNA: 256 occupancy sweet spot.
    /// Fallback: 128 (safe universal default).
    #[must_use]
    pub fn preferred_workgroup_size(&self) -> u32 {
        match self.arch {
            GpuArch::Volta | GpuArch::Turing => 64,
            GpuArch::Ampere | GpuArch::Ada => 256,
            GpuArch::Rdna2 | GpuArch::Rdna3 => 256,
            _ => 128,
        }
    }

    /// Subgroup-aware preferred workgroup size.
    ///
    /// Ensures the workgroup size is a multiple of the actual subgroup
    /// (warp/wavefront) width reported by the adapter. When the subgroup
    /// size is unknown (`None`), falls back to `preferred_workgroup_size()`.
    ///
    /// toadStool S97 discovered that misaligned workgroup sizes cause
    /// partial-subgroup waste on AMD RDNA (wavefront 32 in wave32 mode)
    /// and underutilization on CDNA2 (wavefront 64).
    #[must_use]
    pub fn preferred_workgroup_size_subgroup(&self, subgroup_size: Option<u32>) -> u32 {
        let base = self.preferred_workgroup_size();
        let Some(sg) = subgroup_size.filter(|&s| s > 0) else {
            return base;
        };
        // Align base to a multiple of the subgroup size, rounding up.
        // Then clamp to a sane maximum (1024 is the Vulkan guaranteed minimum
        // for max_compute_invocations_per_workgroup).
        let aligned = base.div_ceil(sg) * sg;
        aligned.min(1024)
    }
}

impl fmt::Display for GpuDriverProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "GPU Driver Profile:")?;
        writeln!(f, "  Driver:   {:?}", self.driver)?;
        writeln!(f, "  Compiler: {:?}", self.compiler)?;
        writeln!(f, "  Arch:     {:?}", self.arch)?;
        writeln!(f, "  FP64:     {:?}", self.fp64_rate)?;
        if !self.workarounds.is_empty() {
            writeln!(f, "  Workarounds: {:?}", self.workarounds)?;
        }
        writeln!(f, "  Eigensolve: {:?}", self.optimal_eigensolve_strategy())?;
        writeln!(f, "  FP64 Strategy: {:?}", self.fp64_strategy())?;
        writeln!(f, "  Precision Routing: {:?}", self.precision_routing())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
