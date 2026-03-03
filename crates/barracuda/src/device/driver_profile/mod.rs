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
//! Reference: `docs/specs/SOVEREIGN_COMPUTE_EVOLUTION.md`

mod architectures;
mod workarounds;

pub use architectures::GpuArch;
pub use workarounds::Workaround;

use std::fmt;

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};

// ── Driver identity ───────────────────────────────────────────────────────────

/// GPU driver/compiler identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriverKind {
    NvidiaProprietary,
    Nvk,
    Radv,
    Intel,
    Software,
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
    WarpPacked { wg_size: u32 },
    /// Wave-packed: 64 independent matrices per workgroup (AMD)
    WavePacked { wave_size: u32 },
    /// Standard: one matrix per workgroup
    Standard,
}

// ── FP64 execution strategy ──────────────────────────────────────────────────

/// Hardware-adaptive FP64 execution strategy (hotSpring core-streaming discovery).
///
/// On compute-class GPUs (Titan V, V100, MI250) with 1:2 FP64:FP32 hardware,
/// native `f64` is fastest. On consumer GPUs (1:64 ratio), routing bulk math
/// through double-float f32-pair (`Df64`) on the massive FP32 core array
/// delivers ~10x the effective throughput, reserving native `f64` for
/// precision-critical reductions and convergence tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp64Strategy {
    /// Use native f64 everywhere (Titan V, V100, A100, MI250X — 1:2 hardware).
    Native,
    /// DF64 (f32-pair, ~14 digits) for bulk math, native f64 for reductions.
    Hybrid,
    /// Run both DF64 and native f64 concurrently, cross-validate results.
    /// Useful for validation harnesses and precision-sensitive pipelines.
    Concurrent,
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
    pub driver: DriverKind,
    pub compiler: CompilerKind,
    pub arch: GpuArch,
    pub fp64_rate: Fp64Rate,
    pub workarounds: Vec<Workaround>,
}

impl GpuDriverProfile {
    /// Build a driver profile from a `WgpuDevice` using runtime detection.
    pub fn from_device(device: &WgpuDevice) -> Self {
        let driver = Self::detect_driver(device);
        let compiler = Self::detect_compiler(driver);
        let arch = architectures::detect_arch(device);
        let fp64_rate = Self::detect_fp64_rate(&arch, driver);
        let workarounds = workarounds::detect_workarounds(driver, arch);

        Self {
            driver,
            compiler,
            arch,
            fp64_rate,
            workarounds,
        }
    }

    /// Optimal eigensolve dispatch strategy for this driver/arch combination.
    ///
    /// hotSpring measured 2.2× NVK speedup with warp-packing (Titan V, Feb 2026).
    /// Neutral on proprietary NVIDIA (scheduler already handles wg1 efficiently).
    ///
    /// ### AMD RDNA2/RDNA3 (ACO compiler)
    ///
    /// Empirically measured on RX 6950 XT (RDNA2/NAVI21, Feb 2026):
    /// - `wg_size=32`: 67.7 ms  ← optimal
    /// - `wg_size=64`: 117.1 ms ← 1.7× slower
    ///
    /// Root cause: ACO targets **wave32 mode** for compute shaders on RDNA2.
    /// Use `WarpPacked { wg_size: 32 }` for all current ACO targets.
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
    pub fn needs_exp_f64_workaround(&self) -> bool {
        self.workarounds.contains(&Workaround::NvkExpF64Crash)
            || self
                .workarounds
                .contains(&Workaround::NvvmAdaF64Transcendentals)
    }

    /// Whether `log(f64)` needs software substitution on this driver.
    pub fn needs_log_f64_workaround(&self) -> bool {
        self.workarounds.contains(&Workaround::NvkLogF64Crash)
            || self
                .workarounds
                .contains(&Workaround::NvvmAdaF64Transcendentals)
    }

    /// Whether `pow(f64, f64)` needs software substitution on this driver.
    ///
    /// Ada Lovelace (SM89) with the proprietary NVIDIA driver cannot compile
    /// native f64 pow/exp/log. Discovered by wetSpring on RTX 4070 (Feb 2026).
    pub fn needs_pow_f64_workaround(&self) -> bool {
        self.workarounds
            .contains(&Workaround::NvvmAdaF64Transcendentals)
    }

    /// Whether `sin(f64)` needs Taylor-series workaround on this driver.
    ///
    /// NVK (open-source NVIDIA Vulkan driver) has broken or imprecise native
    /// sin/cos implementations for f64. Returns true for NVK and any driver
    /// with the `NvkSinCosF64Imprecise` workaround.
    pub fn needs_sin_f64_workaround(&self) -> bool {
        self.workarounds
            .contains(&Workaround::NvkSinCosF64Imprecise)
    }

    /// Whether `cos(f64)` needs Taylor-series workaround on this driver.
    ///
    /// NVK has broken or imprecise native cos for f64. Returns true for NVK
    /// and any driver with the `NvkSinCosF64Imprecise` workaround.
    pub fn needs_cos_f64_workaround(&self) -> bool {
        self.workarounds
            .contains(&Workaround::NvkSinCosF64Imprecise)
    }

    /// Whether this driver supports f64 builtins (exp, log, pow) natively.
    ///
    /// Returns false for NVK, software renderers, and NVIDIA Ada Lovelace
    /// proprietary (NVVM PTXAS fails on f64 transcendentals for SM89).
    pub fn supports_f64_builtins(&self) -> bool {
        self.workarounds.is_empty() && !matches!(self.driver, DriverKind::Software)
    }

    /// Choose the optimal FP64 execution strategy for this hardware.
    ///
    /// Compute-class GPUs (1:2 FP64:FP32) should use native f64 everywhere.
    /// Consumer GPUs (1:64) benefit from the hybrid core-streaming approach:
    /// DF64 (f32-pair) on the FP32 core array for bulk math, native f64 only
    /// for accumulations and convergence tests.
    pub fn fp64_strategy(&self) -> Fp64Strategy {
        match self.fp64_rate {
            Fp64Rate::Full => Fp64Strategy::Native,
            Fp64Rate::Throttled | Fp64Rate::Minimal | Fp64Rate::Software => Fp64Strategy::Hybrid,
        }
    }

    /// Probe-informed FP64 strategy. Overrides heuristic when the runtime
    /// probe shows f64 compilation actually fails (NAK, NVVM).
    ///
    /// groundSpring V35/V37 discovery: NAK and NVVM advertise `SHADER_F64`
    /// but cannot compile f64 WGSL. The probe provides ground truth.
    pub fn fp64_strategy_probed(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> Fp64Strategy {
        if !caps.can_compile_f64() {
            return Fp64Strategy::Hybrid;
        }
        self.fp64_strategy()
    }

    /// Whether `sin(f64)` needs software substitution, considering probe results.
    pub fn needs_sin_f64_workaround_probed(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> bool {
        caps.needs_sin_f64_workaround()
    }

    /// Whether `cos(f64)` needs software substitution, considering probe results.
    pub fn needs_cos_f64_workaround_probed(
        &self,
        caps: &crate::device::probe::F64BuiltinCapabilities,
    ) -> bool {
        caps.needs_cos_f64_workaround()
    }

    /// Whether this is an open-source driver (NVK or RADV).
    pub fn is_open_source(&self) -> bool {
        matches!(self.driver, DriverKind::Nvk | DriverKind::Radv)
    }

    /// Return the `LatencyModel` appropriate for this GPU architecture.
    ///
    /// The model provides per-operation cycle counts used by the WGSL ILP
    /// scheduler (`@ilp_region` reorderer, Phase 3 `WgslDependencyGraph`).
    ///
    /// - NVIDIA Volta/Turing/Ampere/Ada → `Sm70LatencyModel` (DFMA = 8 cy)
    /// - AMD RDNA2/RDNA3/CDNA2 → `Rdna2LatencyModel` (VFMA64 ≈ 4 cy)
    /// - Unknown/Intel/Software → `ConservativeModel` (safe overestimate)
    #[must_use]
    pub fn latency_model(&self) -> Box<dyn crate::device::latency::LatencyModel> {
        crate::device::latency::model_for_arch(self.arch)
    }

    // ── Allocation safety ─────────────────────────────────────────────────────

    /// Maximum safe combined allocation in bytes, or `None` if unlimited.
    ///
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
    /// driver. Returns `Ok(())` if safe, or `Err(DeviceLimitExceeded)` with a
    /// diagnostic message suggesting Mesa git HEAD.
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

    fn detect_fp64_rate(arch: &GpuArch, driver: DriverKind) -> Fp64Rate {
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
    ///
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_profile(rate: Fp64Rate, arch: GpuArch) -> GpuDriverProfile {
        GpuDriverProfile {
            driver: DriverKind::NvidiaProprietary,
            compiler: CompilerKind::NvidiaPtxas,
            arch,
            fp64_rate: rate,
            workarounds: vec![],
        }
    }

    #[test]
    fn fp64_strategy_native_for_full_rate() {
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);
        assert_eq!(p.fp64_strategy(), Fp64Strategy::Native);
    }

    #[test]
    fn fp64_strategy_hybrid_for_throttled() {
        let p = make_profile(Fp64Rate::Throttled, GpuArch::Ampere);
        assert_eq!(p.fp64_strategy(), Fp64Strategy::Hybrid);
    }

    #[test]
    fn fp64_strategy_hybrid_for_minimal() {
        let p = make_profile(Fp64Rate::Minimal, GpuArch::Ada);
        assert_eq!(p.fp64_strategy(), Fp64Strategy::Hybrid);
    }

    #[test]
    fn fp64_strategy_hybrid_for_software() {
        let p = make_profile(Fp64Rate::Software, GpuArch::Software);
        assert_eq!(p.fp64_strategy(), Fp64Strategy::Hybrid);
    }

    #[test]
    fn nvk_allocation_guard_rejects_large() {
        let p = GpuDriverProfile {
            driver: DriverKind::Nvk,
            compiler: CompilerKind::Nak,
            arch: GpuArch::Volta,
            fp64_rate: Fp64Rate::Full,
            workarounds: vec![Workaround::NvkLargeBufferLimit],
        };
        assert!(p.max_safe_total_allocation().is_some());
        assert!(p.check_allocation_safe(500_000_000).is_ok());
        assert!(p.check_allocation_safe(1_500_000_000).is_err());
    }

    #[test]
    fn non_nvk_allocation_guard_allows_any() {
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);
        assert!(p.max_safe_total_allocation().is_none());
        assert!(p.check_allocation_safe(10_000_000_000).is_ok());
    }

    #[test]
    fn display_includes_fp64_strategy() {
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);
        let s = format!("{p}");
        assert!(
            s.contains("FP64 Strategy: Native"),
            "display should show strategy"
        );
    }

    #[test]
    fn fp64_strategy_probed_overrides_when_basic_f64_fails() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);
        assert_eq!(p.fp64_strategy(), Fp64Strategy::Native);

        let caps_no_f64 = F64BuiltinCapabilities::none();
        assert_eq!(
            p.fp64_strategy_probed(&caps_no_f64),
            Fp64Strategy::Hybrid,
            "probe failure must force Hybrid even on Full-rate hardware"
        );

        let caps_full = F64BuiltinCapabilities::full();
        assert_eq!(
            p.fp64_strategy_probed(&caps_full),
            Fp64Strategy::Native,
            "probe success on Full-rate should keep Native"
        );
    }

    #[test]
    fn fp64_strategy_probed_respects_rate_when_probe_passes() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Throttled, GpuArch::Ampere);
        let caps_full = F64BuiltinCapabilities::full();
        assert_eq!(
            p.fp64_strategy_probed(&caps_full),
            Fp64Strategy::Hybrid,
            "Throttled hardware should stay Hybrid even when probe passes"
        );
    }

    #[test]
    fn sin_cos_workaround_probed() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);

        let caps_no_sin = F64BuiltinCapabilities {
            basic_f64: true,
            sin: false,
            cos: true,
            ..F64BuiltinCapabilities::full()
        };
        assert!(p.needs_sin_f64_workaround_probed(&caps_no_sin));
        assert!(!p.needs_cos_f64_workaround_probed(&caps_no_sin));
    }

    #[test]
    fn needs_cos_f64_workaround_probed_when_cos_fails() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);

        let caps_no_cos = F64BuiltinCapabilities {
            basic_f64: true,
            sin: true,
            cos: false,
            ..F64BuiltinCapabilities::full()
        };
        assert!(!p.needs_sin_f64_workaround_probed(&caps_no_cos));
        assert!(p.needs_cos_f64_workaround_probed(&caps_no_cos));
    }

    #[test]
    fn needs_sin_cos_workaround_probed_both_fail() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);

        let caps_none = F64BuiltinCapabilities::none();
        assert!(p.needs_sin_f64_workaround_probed(&caps_none));
        assert!(p.needs_cos_f64_workaround_probed(&caps_none));
    }

    #[test]
    fn needs_sin_cos_workaround_probed_both_ok() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);

        let caps_full = F64BuiltinCapabilities::full();
        assert!(!p.needs_sin_f64_workaround_probed(&caps_full));
        assert!(!p.needs_cos_f64_workaround_probed(&caps_full));
    }

    #[test]
    fn needs_sin_f64_workaround_true_for_nvk() {
        let p = GpuDriverProfile {
            driver: DriverKind::Nvk,
            compiler: CompilerKind::Nak,
            arch: GpuArch::Volta,
            fp64_rate: Fp64Rate::Full,
            workarounds: vec![Workaround::NvkSinCosF64Imprecise],
        };
        assert!(p.needs_sin_f64_workaround());
    }

    #[test]
    fn needs_cos_f64_workaround_true_for_nvk() {
        let p = GpuDriverProfile {
            driver: DriverKind::Nvk,
            compiler: CompilerKind::Nak,
            arch: GpuArch::Volta,
            fp64_rate: Fp64Rate::Full,
            workarounds: vec![Workaround::NvkSinCosF64Imprecise],
        };
        assert!(p.needs_cos_f64_workaround());
    }

    #[test]
    fn needs_sin_cos_f64_workaround_false_for_proprietary_nvidia() {
        let p = make_profile(Fp64Rate::Full, GpuArch::Volta);
        assert!(!p.needs_sin_f64_workaround());
        assert!(!p.needs_cos_f64_workaround());
    }

    #[test]
    fn fp64_strategy_probed_hybrid_when_probe_fails_on_full_rate() {
        use crate::device::probe::F64BuiltinCapabilities;
        let p = make_profile(Fp64Rate::Full, GpuArch::Cdna2);
        assert_eq!(p.fp64_strategy(), Fp64Strategy::Native);

        let caps_fail = F64BuiltinCapabilities::none();
        assert_eq!(
            p.fp64_strategy_probed(&caps_fail),
            Fp64Strategy::Hybrid,
            "Probe failure must override Full rate to Hybrid"
        );
    }
}
