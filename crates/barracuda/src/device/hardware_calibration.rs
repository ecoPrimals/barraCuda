// SPDX-License-Identifier: AGPL-3.0-or-later

//! Hardware calibration: safe per-tier compilation probe.
//!
//! Absorbed from hotSpring v0.6.25. The RTX 3090's proprietary NVIDIA driver
//! demonstrated that a single failed DF64 compilation (NVVM error) permanently
//! poisons the wgpu device — all subsequent buffer and dispatch operations fail.
//! The calibration module probes each tier before routing, using `catch_unwind`
//! to contain failures.
//!
//! # Usage
//!
//! ```rust,ignore
//! let device = WgpuDevice::new().await?;
//! let cal = HardwareCalibration::from_device(&device);
//! assert!(cal.tier_safe(PrecisionTier::F32));
//! println!("{cal}");
//! ```

use super::WgpuDevice;
use super::driver_profile::GpuDriverProfile;
use super::precision_tier::PrecisionTier;

/// Calibrated capability of a single precision tier on this GPU.
#[derive(Debug, Clone)]
pub struct TierCapability {
    /// Which tier was probed.
    pub tier: PrecisionTier,
    /// Whether a minimal shader compiled at this tier.
    pub compiles: bool,
    /// Whether dispatch + readback produced valid (non-NaN, non-zero) output.
    pub dispatches: bool,
    /// Whether transcendental builtins (exp, log) are safe at this tier.
    /// `false` on NVIDIA proprietary for DF64 transcendentals (NVVM poisoning).
    pub transcendentals_safe: bool,
}

/// Complete hardware calibration for a single GPU.
///
/// Built once at startup via [`from_device`](Self::from_device) or
/// [`from_profile`](Self::from_profile). All subsequent routing queries
/// are O(1) lookups into the tier capability array.
#[derive(Debug, Clone)]
pub struct HardwareCalibration {
    /// Adapter name for diagnostics.
    pub adapter_name: String,
    /// Per-tier probe results: `F32`, `DF64`, `F64`, `F64Precise`.
    pub tiers: Vec<TierCapability>,
    /// Whether any f64-class tier is available (`F64` or `F64Precise`).
    pub has_any_f64: bool,
    /// Whether DF64 compilation works (critical for consumer GPUs).
    pub df64_safe: bool,
    /// Whether any tier has NVVM transcendental issues.
    pub nvvm_transcendental_risk: bool,
}

impl HardwareCalibration {
    /// Build calibration from a `WgpuDevice` using the existing probe infrastructure.
    ///
    /// This is lighter weight than hotSpring's full GPU dispatch probe: it
    /// synthesizes tier safety from the driver profile and cached probe results
    /// rather than dispatching test shaders at each tier. This avoids the risk
    /// of probe-induced device poisoning on NVIDIA proprietary.
    #[must_use]
    pub fn from_device(device: &WgpuDevice) -> Self {
        let profile = GpuDriverProfile::from_device(device);
        Self::from_profile(&profile, device)
    }

    /// Build calibration from a pre-computed driver profile.
    #[must_use]
    pub fn from_profile(profile: &GpuDriverProfile, device: &WgpuDevice) -> Self {
        let adapter_name = device.name().to_string();
        let has_f64_shaders = device.has_f64_shaders();
        let has_nvvm_risk = profile.has_df64_spir_v_poisoning();
        let precision_routing = profile.precision_routing();

        use super::driver_profile::PrecisionRoutingAdvice as PRA;

        let has_f16 = device
            .device()
            .features()
            .contains(wgpu::Features::SHADER_F16);

        let f16_cap = TierCapability {
            tier: PrecisionTier::F16,
            compiles: has_f16,
            dispatches: has_f16,
            transcendentals_safe: has_f16,
        };

        let f32_cap = TierCapability {
            tier: PrecisionTier::F32,
            compiles: true,
            dispatches: true,
            transcendentals_safe: true,
        };

        let df64_compiles = !matches!(precision_routing, PRA::F32Only);
        let df64_transcendentals = df64_compiles && !has_nvvm_risk;
        let df64_cap = TierCapability {
            tier: PrecisionTier::DF64,
            compiles: df64_compiles,
            dispatches: df64_compiles,
            transcendentals_safe: df64_transcendentals,
        };

        let f64_works = has_f64_shaders
            && matches!(
                precision_routing,
                PRA::F64Native | PRA::F64NativeNoSharedMem
            );
        let f64_cap = TierCapability {
            tier: PrecisionTier::F64,
            compiles: f64_works,
            dispatches: f64_works,
            transcendentals_safe: f64_works && profile.supports_f64_builtins(),
        };

        let f64_precise_cap = TierCapability {
            tier: PrecisionTier::F64Precise,
            compiles: f64_works,
            dispatches: f64_works,
            transcendentals_safe: f64_works && profile.supports_f64_builtins(),
        };

        let tiers = vec![f16_cap, f32_cap, df64_cap, f64_cap, f64_precise_cap];

        let has_any_f64 = tiers.iter().any(|t| {
            t.dispatches && matches!(t.tier, PrecisionTier::F64 | PrecisionTier::F64Precise)
        });
        let df64_safe = tiers
            .iter()
            .any(|t| t.dispatches && t.tier == PrecisionTier::DF64);
        let nvvm_transcendental_risk = tiers
            .iter()
            .any(|t| t.dispatches && !t.transcendentals_safe);

        Self {
            adapter_name,
            tiers,
            has_any_f64,
            df64_safe,
            nvvm_transcendental_risk,
        }
    }

    /// Look up a specific tier's capability.
    #[must_use]
    pub fn tier_cap(&self, tier: PrecisionTier) -> Option<&TierCapability> {
        self.tiers.iter().find(|t| t.tier == tier)
    }

    /// Check whether a tier is safe to use (compiles + dispatches + transcendentals).
    #[must_use]
    pub fn tier_safe(&self, tier: PrecisionTier) -> bool {
        self.tier_cap(tier)
            .is_some_and(|t| t.compiles && t.dispatches && t.transcendentals_safe)
    }

    /// Check whether a tier can dispatch arithmetic-only shaders
    /// (no exp/log/transcendentals). Useful for DF64 on proprietary NVIDIA.
    #[must_use]
    pub fn tier_arith_only(&self, tier: PrecisionTier) -> bool {
        self.tier_cap(tier)
            .is_some_and(|t| t.compiles && t.dispatches && !t.transcendentals_safe)
    }

    /// Best safe tier for f64-class work. Prefers native F64, falls back
    /// through `F64Precise` → `DF64`.
    #[must_use]
    pub fn best_f64_tier(&self) -> Option<PrecisionTier> {
        [
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::DF64,
        ]
        .into_iter()
        .find(|&t| self.tier_safe(t))
    }

    /// Best safe tier overall (including F32 as fallback).
    #[must_use]
    pub fn best_any_tier(&self) -> Option<PrecisionTier> {
        [
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::DF64,
            PrecisionTier::F32,
        ]
        .into_iter()
        .find(|&t| self.tier_safe(t))
    }
}

impl std::fmt::Display for HardwareCalibration {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "HwCal[{}]:", self.adapter_name)?;
        for t in &self.tiers {
            let mark = if t.dispatches && t.transcendentals_safe {
                "\u{2713}" // ✓
            } else if t.dispatches {
                "\u{25b3}arith" // △arith
            } else if t.compiles {
                "\u{25b3}comp" // △comp
            } else {
                "\u{2717}" // ✗
            };
            write!(f, " {}={mark}", t.tier)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn tier_safe_queries() {
        let cal = make_cal(true, true, true, false);
        assert!(cal.tier_safe(PrecisionTier::F32));
        assert!(cal.tier_safe(PrecisionTier::DF64));
        assert!(cal.tier_safe(PrecisionTier::F64));
        assert!(!cal.tier_safe(PrecisionTier::F64Precise));
    }

    #[test]
    fn best_f64_tier_preference() {
        let cal = make_cal(true, true, true, true);
        assert_eq!(cal.best_f64_tier(), Some(PrecisionTier::F64));

        let cal = make_cal(true, true, false, false);
        assert_eq!(cal.best_f64_tier(), Some(PrecisionTier::DF64));

        let cal = make_cal(true, false, false, false);
        assert_eq!(cal.best_f64_tier(), None);
    }

    #[test]
    fn best_any_tier_fallback() {
        let cal = make_cal(true, false, false, false);
        assert_eq!(cal.best_any_tier(), Some(PrecisionTier::F32));
    }

    #[test]
    fn arith_only_detection() {
        let mut cal = make_cal(true, true, true, true);
        if let Some(df64) = cal.tiers.iter_mut().find(|t| t.tier == PrecisionTier::DF64) {
            df64.transcendentals_safe = false;
        }
        assert!(cal.tier_arith_only(PrecisionTier::DF64));
        assert!(!cal.tier_arith_only(PrecisionTier::F64));
    }

    #[test]
    fn display_format() {
        let cal = make_cal(true, true, true, false);
        let s = cal.to_string();
        assert!(s.contains("F64=\u{2713}"), "F64 should show ✓: {s}");
        assert!(
            s.contains("F64Precise=\u{2717}"),
            "F64Precise should show ✗: {s}"
        );
    }
}
