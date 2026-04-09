// SPDX-License-Identifier: AGPL-3.0-or-later

//! Self-routing precision brain — absorbed from hotSpring v0.6.25.
//!
//! Routes physics workloads to the best available precision tier, never
//! touching a tier that failed probing. Probe-first, data-driven,
//! domain-aware.
//!
//! # Design principles
//!
//! 1. **Probe-first**: Never assume a tier works. The RTX 3090 NVVM failure
//!    demonstrated that a single bad DF64 compilation poisons the wgpu device.
//! 2. **Data-driven**: Routing decisions from calibration data, not static
//!    driver-name heuristics.
//! 3. **Domain-aware**: Physics requirements (FMA sensitivity, precision floor)
//!    combined with hardware capability.
//! 4. **Portable**: Works in any spring that can construct a `WgpuDevice`.
//!
//! # Usage
//!
//! ```rust,ignore
//! let device = WgpuDevice::new().await?;
//! let brain = PrecisionBrain::from_device(&device);
//! let tier = brain.route(PhysicsDomain::Dielectric);
//! let module = brain.compile(&device, PhysicsDomain::LatticeQcd, &source, "my_shader");
//! ```

use super::WgpuDevice;
use super::capabilities::DeviceCapabilities;
use super::driver_profile::PrecisionRoutingAdvice as HwAdvice;
use super::hardware_calibration::HardwareCalibration;
use super::precision_tier::{PhysicsDomain, PrecisionBrainAdvice, PrecisionTier};

/// Number of registered domains for the routing table.
const NUM_DOMAINS: usize = 15;

const ALL_DOMAINS: [PhysicsDomain; NUM_DOMAINS] = [
    PhysicsDomain::LatticeQcd,
    PhysicsDomain::GradientFlow,
    PhysicsDomain::Dielectric,
    PhysicsDomain::KineticFluid,
    PhysicsDomain::Eigensolve,
    PhysicsDomain::MolecularDynamics,
    PhysicsDomain::NuclearEos,
    PhysicsDomain::PopulationPk,
    PhysicsDomain::Bioinformatics,
    PhysicsDomain::Hydrology,
    PhysicsDomain::Statistics,
    PhysicsDomain::General,
    PhysicsDomain::Inference,
    PhysicsDomain::Training,
    PhysicsDomain::Hashing,
];

/// Self-routing precision brain for a single GPU.
///
/// Constructed once at startup. All subsequent routing calls are O(1) lookups.
pub struct PrecisionBrain {
    /// The calibration data from probing.
    pub calibration: HardwareCalibration,
    route_table: [PrecisionTier; NUM_DOMAINS],
    hw_native: bool,
    /// Whether coralReef sovereign compilation is available for f64 lowering.
    /// When true, F64 tiers are accessible even when hardware native
    /// transcendentals fail — coralReef provides software polyfills in the
    /// compiled native binary.
    coral_f64_lowering: bool,
}

impl PrecisionBrain {
    /// Build the brain from a `WgpuDevice`, probing hardware and constructing
    /// the routing table.
    #[must_use]
    pub fn from_device(device: &WgpuDevice) -> Self {
        let caps = DeviceCapabilities::from_device(device);
        let calibration = HardwareCalibration::from_capabilities(&caps, device);
        Self::from_capabilities(calibration, &caps)
    }

    /// Build the brain with coralReef f64 lowering awareness.
    ///
    /// When `coral_f64_lowering` is true, the brain routes more aggressively
    /// to F64/DF64 tiers because coralReef provides software polyfills for
    /// transcendentals that hardware probes report as broken. The sovereign
    /// compilation pipeline bypasses naga/NVVM, so driver bugs are irrelevant.
    #[must_use]
    pub fn from_device_with_coral(device: &WgpuDevice, coral_f64_lowering: bool) -> Self {
        let caps = DeviceCapabilities::from_device(device);
        let calibration = HardwareCalibration::from_capabilities(&caps, device);
        Self::from_capabilities_with_coral(calibration, &caps, coral_f64_lowering)
    }

    /// Build the brain from pre-computed calibration and capabilities.
    #[must_use]
    pub fn from_capabilities(calibration: HardwareCalibration, caps: &DeviceCapabilities) -> Self {
        Self::from_capabilities_with_coral(calibration, caps, false)
    }

    /// Build the brain with explicit coralReef lowering flag.
    #[must_use]
    pub fn from_capabilities_with_coral(
        calibration: HardwareCalibration,
        caps: &DeviceCapabilities,
        coral_f64_lowering: bool,
    ) -> Self {
        let hw_native = matches!(
            caps.precision_routing(),
            HwAdvice::F64Native | HwAdvice::F64NativeNoSharedMem
        );

        let route_table = ALL_DOMAINS
            .map(|domain| route_domain(domain, &calibration, hw_native, coral_f64_lowering));

        tracing::info!(
            "PrecisionBrain[{}]: {calibration} (coral_f64_lowering={coral_f64_lowering})",
            calibration.adapter_name
        );
        for (i, domain) in ALL_DOMAINS.iter().enumerate() {
            tracing::debug!("  {domain:?} → {:?}", route_table[i]);
        }

        Self {
            calibration,
            route_table,
            hw_native,
            coral_f64_lowering,
        }
    }

    /// Route a physics domain to the best available precision tier.
    /// O(1) lookup.
    #[must_use]
    pub fn route(&self, domain: PhysicsDomain) -> PrecisionTier {
        self.route_table[domain_index(domain)]
    }

    /// Route and return full advice (tier + rationale + FMA safety).
    #[must_use]
    pub fn route_advice(&self, domain: PhysicsDomain) -> PrecisionBrainAdvice {
        let tier = self.route(domain);
        let (fma_safe, rationale) = domain_requirements(domain, tier);
        PrecisionBrainAdvice {
            tier,
            fma_safe,
            rationale,
        }
    }

    /// Compile a shader at the brain-selected tier for the given domain.
    ///
    /// Scale-down tiers (Binary through Bf16) and F32 use the universal
    /// `compile_shader` path — compute is in f32 after dequantization.
    /// Scale-up tiers route to their respective compilation paths.
    /// QF128 uses the f32 path (all f32 arithmetic, no f64 hardware needed).
    /// DF128 uses the f64 path (Dekker double-double on f64).
    #[must_use]
    pub fn compile(
        &self,
        device: &WgpuDevice,
        domain: PhysicsDomain,
        shader_source: &str,
        label: &str,
    ) -> wgpu::ShaderModule {
        let tier = self.route(domain);
        match tier {
            PrecisionTier::Binary
            | PrecisionTier::Int2
            | PrecisionTier::Quantized4
            | PrecisionTier::Quantized8
            | PrecisionTier::Fp8E5M2
            | PrecisionTier::Fp8E4M3
            | PrecisionTier::Bf16
            | PrecisionTier::F16
            | PrecisionTier::Tf32
            | PrecisionTier::F32
            | PrecisionTier::QF128 => device.compile_shader(shader_source, Some(label)),
            PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => {
                device.compile_shader_f64(shader_source, Some(label))
            }
            PrecisionTier::DF64 => device.compile_shader_df64(shader_source, Some(label)),
        }
    }

    /// Check if a specific tier is safe on this hardware.
    #[must_use]
    pub fn tier_safe(&self, tier: PrecisionTier) -> bool {
        self.calibration.tier_safe(tier)
    }

    /// Whether this hardware has native f64 capability.
    #[must_use]
    pub fn has_native_f64(&self) -> bool {
        self.hw_native
    }

    /// Whether coralReef sovereign f64 lowering is available.
    #[must_use]
    pub fn has_coral_f64_lowering(&self) -> bool {
        self.coral_f64_lowering
    }

    /// Adapter name.
    #[must_use]
    pub fn adapter_name(&self) -> &str {
        &self.calibration.adapter_name
    }

    /// Whether the routed tier for this domain requires sovereign compilation.
    ///
    /// Returns `true` when the domain routes to an f64-class tier and the
    /// hardware doesn't natively support it, but coralReef lowering
    /// makes it viable. In this case, the caller should prefer the coralReef
    /// `compile_wgsl_direct` path over the wgpu/naga path.
    #[must_use]
    pub fn needs_sovereign_compile(&self, domain: PhysicsDomain) -> bool {
        if !self.coral_f64_lowering {
            return false;
        }
        let tier = self.route(domain);
        matches!(
            tier,
            PrecisionTier::F64
                | PrecisionTier::F64Precise
                | PrecisionTier::DF64
                | PrecisionTier::DF128
        ) && !self.hw_native
    }
}

impl std::fmt::Display for PrecisionBrain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PrecisionBrain[{}]:", self.calibration.adapter_name)?;
        for (i, domain) in ALL_DOMAINS.iter().enumerate() {
            writeln!(f, "  {domain:?} → {:?}", self.route_table[i])?;
        }
        Ok(())
    }
}

// ── Routing table construction ──────────────────────────────────────────────

const fn domain_index(domain: PhysicsDomain) -> usize {
    match domain {
        PhysicsDomain::LatticeQcd => 0,
        PhysicsDomain::GradientFlow => 1,
        PhysicsDomain::Dielectric => 2,
        PhysicsDomain::KineticFluid => 3,
        PhysicsDomain::Eigensolve => 4,
        PhysicsDomain::MolecularDynamics => 5,
        PhysicsDomain::NuclearEos => 6,
        PhysicsDomain::PopulationPk => 7,
        PhysicsDomain::Bioinformatics => 8,
        PhysicsDomain::Hydrology => 9,
        PhysicsDomain::Statistics => 10,
        PhysicsDomain::General => 11,
        PhysicsDomain::Inference => 12,
        PhysicsDomain::Training => 13,
        PhysicsDomain::Hashing => 14,
    }
}

fn route_domain(
    domain: PhysicsDomain,
    cal: &HardwareCalibration,
    hw_native: bool,
    coral_f64_lowering: bool,
) -> PrecisionTier {
    let f64_safe = cal.tier_safe(PrecisionTier::F64) || coral_f64_lowering;
    let df64_safe = cal.tier_safe(PrecisionTier::DF64) || coral_f64_lowering;
    let precise_safe = cal.tier_safe(PrecisionTier::F64Precise) || coral_f64_lowering;
    let f16_safe = cal.tier_safe(PrecisionTier::F16);
    let bf16_safe = cal.tier_safe(PrecisionTier::Bf16);

    match domain {
        // Precision-critical: prefer F64Precise, cascade through tiers
        PhysicsDomain::Dielectric | PhysicsDomain::Eigensolve => {
            if (hw_native || coral_f64_lowering) && precise_safe {
                PrecisionTier::F64Precise
            } else if f64_safe {
                PrecisionTier::F64
            } else if df64_safe {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }

        // Moderate precision: prefer F64, cascade to DF64
        PhysicsDomain::GradientFlow
        | PhysicsDomain::NuclearEos
        | PhysicsDomain::PopulationPk
        | PhysicsDomain::Hydrology => {
            if f64_safe {
                PrecisionTier::F64
            } else if df64_safe {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }

        // Throughput-bound: F64 if fast, else DF64 for throughput
        PhysicsDomain::LatticeQcd
        | PhysicsDomain::KineticFluid
        | PhysicsDomain::MolecularDynamics
        | PhysicsDomain::Bioinformatics
        | PhysicsDomain::Statistics
        | PhysicsDomain::General => {
            if f64_safe {
                PrecisionTier::F64
            } else if df64_safe {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }

        // Inference: maximize throughput with quantized/reduced precision.
        // Q4 → Q8 → FP8 → F16 → BF16 → F32 cascade.
        PhysicsDomain::Inference => {
            if cal.tier_safe(PrecisionTier::Quantized4) {
                PrecisionTier::Quantized4
            } else if cal.tier_safe(PrecisionTier::Quantized8) {
                PrecisionTier::Quantized8
            } else if cal.tier_safe(PrecisionTier::Fp8E4M3) {
                PrecisionTier::Fp8E4M3
            } else if f16_safe {
                PrecisionTier::F16
            } else if bf16_safe {
                PrecisionTier::Bf16
            } else {
                PrecisionTier::F32
            }
        }

        // Training: mixed precision with accumulation in f32.
        // BF16 → F16 → F32 → DF64 cascade.
        PhysicsDomain::Training => {
            if bf16_safe {
                PrecisionTier::Bf16
            } else if f16_safe {
                PrecisionTier::F16
            } else if df64_safe {
                PrecisionTier::DF64
            } else {
                PrecisionTier::F32
            }
        }

        // Hashing: binary XNOR+popcount.
        PhysicsDomain::Hashing => PrecisionTier::Binary,
    }
}

fn domain_requirements(domain: PhysicsDomain, tier: PrecisionTier) -> (bool, &'static str) {
    if domain.fma_sensitive() {
        return match tier {
            PrecisionTier::F64Precise => (
                false,
                "FMA-free f64: cancellation-safe for complex arithmetic",
            ),
            PrecisionTier::F64 | PrecisionTier::DF128 => (
                true,
                "Native f64 (FMA-free unavailable, acceptable precision)",
            ),
            PrecisionTier::DF64 | PrecisionTier::QF128 => (
                false,
                "Emulated precision: ~14+ digits, sufficient for most physics",
            ),
            _ => (true, "Reduced precision fallback: validation recommended"),
        };
    }

    if domain.throughput_bound() {
        return match tier {
            PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => {
                (true, "Native f64 for compute-bound domains")
            }
            PrecisionTier::DF64 | PrecisionTier::QF128 => (
                true,
                "Emulated precision throughput mode: f32 cores for max dispatch rate",
            ),
            PrecisionTier::Binary => (true, "Binary XNOR+popcount: maximum throughput"),
            PrecisionTier::Int2 | PrecisionTier::Quantized4 | PrecisionTier::Quantized8 => {
                (true, "Quantized fast path: dequantize→f32 compute")
            }
            PrecisionTier::Fp8E4M3 | PrecisionTier::Fp8E5M2 => {
                (true, "FP8 fast path: 4× throughput over f32")
            }
            PrecisionTier::Bf16 => (true, "BF16 mixed precision: training fast path"),
            PrecisionTier::F16 => (true, "F16 tensor core fast path: max throughput"),
            PrecisionTier::Tf32 => (true, "TF32 tensor core: f32 range, reduced mantissa"),
            PrecisionTier::F32 => (true, "F32 screening/preview mode"),
        };
    }

    match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => {
            (true, "Native f64 for moderate precision needs")
        }
        PrecisionTier::DF64 | PrecisionTier::QF128 => {
            (true, "Emulated precision provides sufficient accuracy")
        }
        PrecisionTier::F16 => (true, "F16 fast path: screening/ML inference"),
        PrecisionTier::Bf16 => (
            true,
            "BF16: ML training, range-preserving reduced precision",
        ),
        PrecisionTier::Fp8E4M3 | PrecisionTier::Fp8E5M2 => (true, "FP8: ultra-low precision ML"),
        PrecisionTier::Binary | PrecisionTier::Int2 => (true, "Integer quantized: exact discrete"),
        PrecisionTier::Quantized4 | PrecisionTier::Quantized8 => {
            (true, "Block quantized: dequantize→compute→quantize")
        }
        PrecisionTier::Tf32 => (true, "TF32: tensor core internal format"),
        PrecisionTier::F32 => (true, "F32 fallback: validate results"),
    }
}

#[cfg(test)]
#[path = "precision_brain_tests.rs"]
mod tests;
