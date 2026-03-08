// SPDX-License-Identifier: AGPL-3.0-or-later

use super::*;

fn make_profile(rate: Fp64Rate, arch: GpuArch) -> GpuDriverProfile {
    GpuDriverProfile {
        driver: DriverKind::NvidiaProprietary,
        compiler: CompilerKind::NvidiaPtxas,
        arch,
        fp64_rate: rate,
        workarounds: vec![],
        adapter_key: String::new(),
    }
}

fn make_profile_with_driver(rate: Fp64Rate, arch: GpuArch, driver: DriverKind) -> GpuDriverProfile {
    GpuDriverProfile {
        driver,
        compiler: CompilerKind::NvidiaPtxas,
        arch,
        fp64_rate: rate,
        workarounds: vec![],
        adapter_key: String::new(),
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
        adapter_key: String::new(),
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
        adapter_key: String::new(),
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
        adapter_key: String::new(),
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

#[test]
fn ada_proprietary_f64_zeros_risk() {
    let p = make_profile(Fp64Rate::Throttled, GpuArch::Ada);
    assert!(
        p.f64_zeros_risk(),
        "Ada + proprietary should report f64 zeros risk"
    );
}

#[test]
fn nvk_f64_zeros_risk() {
    let p = make_profile_with_driver(Fp64Rate::Full, GpuArch::Volta, DriverKind::Nvk);
    assert!(p.f64_zeros_risk(), "NVK should report f64 zeros risk");
}

#[test]
fn ampere_proprietary_no_f64_zeros_risk() {
    let p = make_profile(Fp64Rate::Throttled, GpuArch::Ampere);
    assert!(
        !p.f64_zeros_risk(),
        "Ampere + proprietary should NOT report f64 zeros risk"
    );
}

#[test]
fn ada_proprietary_precision_routing_no_shared_mem() {
    let p = make_profile(Fp64Rate::Throttled, GpuArch::Ada);
    assert_eq!(
        p.precision_routing(),
        PrecisionRoutingAdvice::F64NativeNoSharedMem,
        "Ada + proprietary should route to F64NativeNoSharedMem"
    );
}
