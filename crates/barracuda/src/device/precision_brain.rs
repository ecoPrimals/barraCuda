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
const NUM_DOMAINS: usize = 12;

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
            PrecisionTier::F16 | PrecisionTier::F32 => {
                device.compile_shader(shader_source, Some(label))
            }
            PrecisionTier::F64 | PrecisionTier::F64Precise => {
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
    /// Returns `true` when the domain routes to F64/F64Precise/DF64 and the
    /// hardware doesn't natively support that tier, but coralReef lowering
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
            PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF64
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
    }
}

fn domain_requirements(domain: PhysicsDomain, tier: PrecisionTier) -> (bool, &'static str) {
    if domain.fma_sensitive() {
        return match tier {
            PrecisionTier::F64Precise => (
                false,
                "FMA-free f64: cancellation-safe for complex arithmetic",
            ),
            PrecisionTier::F64 => (
                true,
                "Native f64 (FMA-free unavailable, acceptable precision)",
            ),
            PrecisionTier::DF64 => (
                false,
                "DF64 emulation: ~14 digits, sufficient for most physics",
            ),
            PrecisionTier::F16 | PrecisionTier::F32 => (
                true,
                "F32/F16 fallback: reduced precision, validation recommended",
            ),
        };
    }

    if domain.throughput_bound() {
        return match tier {
            PrecisionTier::F64 | PrecisionTier::F64Precise => {
                (true, "Native f64 for compute-bound domains")
            }
            PrecisionTier::DF64 => (
                true,
                "DF64 throughput mode: f32 cores for max dispatch rate",
            ),
            PrecisionTier::F16 => (true, "F16 tensor core fast path: max throughput"),
            PrecisionTier::F32 => (true, "F32 screening/preview mode"),
        };
    }

    match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise => {
            (true, "Native f64 for moderate precision needs")
        }
        PrecisionTier::DF64 => (true, "DF64 provides sufficient precision"),
        PrecisionTier::F16 => (true, "F16 fast path: screening/ML inference"),
        PrecisionTier::F32 => (true, "F32 fallback: validate results"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::capabilities::DeviceCapabilities;
    use crate::device::hardware_calibration::TierCapability;
    use crate::device::vendor::VENDOR_NVIDIA;

    #[expect(
        clippy::fn_params_excessive_bools,
        reason = "test helper — bool flags are clear and concise here"
    )]
    fn make_cal(
        f32_ok: bool,
        df64_ok: bool,
        f64_ok: bool,
        precise_ok: bool,
    ) -> HardwareCalibration {
        let mk = |tier, ok: bool| TierCapability {
            tier,
            compiles: ok,
            dispatches: ok,
            transcendentals_safe: ok,
        };
        HardwareCalibration {
            adapter_name: "Test GPU".into(),
            tiers: vec![
                mk(PrecisionTier::F32, f32_ok),
                mk(PrecisionTier::DF64, df64_ok),
                mk(PrecisionTier::F64, f64_ok),
                mk(PrecisionTier::F64Precise, precise_ok),
            ],
            has_any_f64: f64_ok || precise_ok,
            df64_safe: df64_ok,
            nvvm_transcendental_risk: false,
        }
    }

    #[test]
    fn full_hw_routes_dielectric_to_precise() {
        let cal = make_cal(true, true, true, true);
        let tier = route_domain(PhysicsDomain::Dielectric, &cal, true, false);
        assert_eq!(tier, PrecisionTier::F64Precise);
    }

    #[test]
    fn no_precise_routes_dielectric_to_f64() {
        let cal = make_cal(true, true, true, false);
        let tier = route_domain(PhysicsDomain::Dielectric, &cal, true, false);
        assert_eq!(tier, PrecisionTier::F64);
    }

    #[test]
    fn no_f64_routes_to_df64() {
        let cal = make_cal(true, true, false, false);
        let tier = route_domain(PhysicsDomain::Dielectric, &cal, false, false);
        assert_eq!(tier, PrecisionTier::DF64);
    }

    #[test]
    fn nothing_works_falls_to_f32() {
        let cal = make_cal(true, false, false, false);
        let tier = route_domain(PhysicsDomain::LatticeQcd, &cal, false, false);
        assert_eq!(tier, PrecisionTier::F32);
    }

    #[test]
    fn throughput_domain_prefers_f64() {
        let cal = make_cal(true, true, true, true);
        let tier = route_domain(PhysicsDomain::MolecularDynamics, &cal, true, false);
        assert_eq!(tier, PrecisionTier::F64);
    }

    #[test]
    fn population_pk_routes_moderate() {
        let cal = make_cal(true, true, true, false);
        let tier = route_domain(PhysicsDomain::PopulationPk, &cal, true, false);
        assert_eq!(tier, PrecisionTier::F64);
    }

    #[test]
    fn hydrology_fallback_to_df64() {
        let cal = make_cal(true, true, false, false);
        let tier = route_domain(PhysicsDomain::Hydrology, &cal, false, false);
        assert_eq!(tier, PrecisionTier::DF64);
    }

    #[test]
    fn domain_index_roundtrip() {
        for (i, &domain) in ALL_DOMAINS.iter().enumerate() {
            assert_eq!(domain_index(domain), i);
        }
    }

    #[test]
    fn advice_fma_sensitive() {
        let (fma_safe, _) =
            domain_requirements(PhysicsDomain::Dielectric, PrecisionTier::F64Precise);
        assert!(!fma_safe);
    }

    #[test]
    fn advice_throughput_bound() {
        let (fma_safe, _) = domain_requirements(PhysicsDomain::LatticeQcd, PrecisionTier::DF64);
        assert!(fma_safe);
    }

    fn test_caps_volta_full() -> DeviceCapabilities {
        DeviceCapabilities {
            device_name: "Test GPU".into(),
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: 1024 * 1024 * 1024,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: wgpu::Backend::Vulkan,
            vendor: VENDOR_NVIDIA,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
            f64_capabilities: None,
        }
    }

    #[test]
    fn domain_requirements_moderate_domains_native_f64() {
        let (fma, rationale) = domain_requirements(PhysicsDomain::GradientFlow, PrecisionTier::F64);
        assert!(fma);
        assert!(rationale.contains("moderate"));
    }

    #[test]
    fn route_advice_dielectric_includes_fma_flag() {
        let cal = make_cal(true, true, true, true);
        let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
        let adv = brain.route_advice(PhysicsDomain::Dielectric);
        assert_eq!(adv.tier, PrecisionTier::F64Precise);
        assert!(!adv.fma_safe);
        assert!(!adv.rationale.is_empty());
    }

    #[test]
    fn precision_brain_display_covers_adapter_name() {
        let cal = make_cal(true, true, true, true);
        let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
        let s = brain.to_string();
        assert!(s.contains("Test GPU"));
        assert!(s.contains("LatticeQcd"));
    }

    #[test]
    fn coral_aware_routes_f64_when_hw_broken_but_coral_available() {
        let cal = make_cal(true, false, false, false);
        let brain =
            PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), true);
        let tier = brain.route(PhysicsDomain::LatticeQcd);
        assert_eq!(
            tier,
            PrecisionTier::F64,
            "with coral lowering, F64 should be routed even without hw native"
        );
    }

    #[test]
    fn coral_aware_routes_precise_for_dielectric() {
        let cal = make_cal(true, true, true, true);
        let brain =
            PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), true);
        let tier = brain.route(PhysicsDomain::Dielectric);
        assert_eq!(tier, PrecisionTier::F64Precise);
    }

    fn test_caps_no_f64() -> DeviceCapabilities {
        DeviceCapabilities {
            f64_shaders: false,
            ..test_caps_volta_full()
        }
    }

    #[test]
    fn needs_sovereign_compile_true_when_coral_and_no_hw() {
        let cal = make_cal(true, false, false, false);
        let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);
        assert!(brain.needs_sovereign_compile(PhysicsDomain::LatticeQcd));
        assert!(brain.has_coral_f64_lowering());
    }

    #[test]
    fn needs_sovereign_compile_false_when_hw_native() {
        let cal = make_cal(true, true, true, true);
        let brain =
            PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), true);
        assert!(
            !brain.needs_sovereign_compile(PhysicsDomain::LatticeQcd),
            "hw native means wgpu path works, no sovereign needed"
        );
    }

    #[test]
    fn needs_sovereign_compile_false_without_coral() {
        let cal = make_cal(true, false, false, false);
        let brain =
            PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), false);
        assert!(!brain.needs_sovereign_compile(PhysicsDomain::LatticeQcd));
        assert!(!brain.has_coral_f64_lowering());
    }

    #[test]
    fn has_native_f64_true_when_profile_reports_native_paths() {
        let cal = make_cal(true, true, true, true);
        let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
        assert!(brain.has_native_f64());
    }

    #[test]
    fn adapter_name_accessor() {
        let cal = make_cal(true, true, true, true);
        let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
        assert_eq!(brain.adapter_name(), "Test GPU");
    }
}
