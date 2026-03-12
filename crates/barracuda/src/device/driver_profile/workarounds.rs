// SPDX-License-Identifier: AGPL-3.0-only
//! Driver workarounds — known bugs and their mitigations.
//!
//! Consolidates workaround detection and allocation limits for drivers
//! that require special handling (NVK PTE faults, NVVM f64 transcendentals, etc.).

use super::{DriverKind, GpuArch};

/// Conservative allocation limit for NVK (nouveau) to avoid kernel PTE fault.
/// Observed on GV100/Titan V: nouveau driver faults above ~1.4 GB combined allocation.
pub(crate) const NVK_MAX_SAFE_ALLOCATION_BYTES: u64 = 1_200_000_000;

/// A known driver/compiler workaround that must be active for a given profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Workaround {
    /// NVK exp(f64) crashes — substitute with polynomial approximation
    NvkExpF64Crash,
    /// NVK log(f64) crashes — substitute with polynomial approximation
    NvkLogF64Crash,
    /// NVIDIA proprietary driver (NVVM/PTXAS) on Ada Lovelace (SM89) fails to
    /// compile native f64 transcendentals (pow, exp, log). Discovered by
    /// wetSpring on RTX 4070 (Feb 2026). Workaround: inject polyfill functions.
    NvvmAdaF64Transcendentals,
    /// NVK (nouveau) PTE fault on large combined allocations (>~1.4 GB).
    /// Conservative limit of 1.2 GB total. File upstream bug in `drm_gpuvm`.
    NvkLargeBufferLimit,
    /// NVK has broken or imprecise sin/cos/tan for f64. Use Taylor series
    /// approximations (`sin_f64_safe`, `cos_f64_safe`) instead of native or polyfill.
    NvkSinCosF64Imprecise,
    /// naga WGSL->SPIR-V codegen poisoning: DF64 shaders containing f64
    /// transcendentals (exp, sqrt, log, pow) produce all-zero output when
    /// compiled through naga's SPIR-V backend. Affects ALL Vulkan drivers
    /// (NVIDIA proprietary, NVK/NAK, llvmpipe) because the root cause is
    /// naga codegen, not the driver JIT.
    ///
    /// On NVIDIA proprietary, the bad SPIR-V additionally causes unrecoverable
    /// NVVM device invalidation ("Buffer is invalid" panic).
    ///
    /// Discovered by hotSpring v0.6.25 on RTX 3090 (Ampere). Confirmed on
    /// Titan V (NVK) and llvmpipe by hotSpring Exp 055 (March 2026).
    /// coralReef Iter 33 validated that sovereign compilation (bypassing naga
    /// SPIR-V) produces correct results for the exact same shaders.
    Df64SpirVPoisoning,
    /// Volta (GV100) lacks PMU firmware for desktop GPUs. nouveau cannot
    /// create compute channels without PMU. When nvPmu (software PMU) has
    /// initialized the compute engine via BAR0 MMIO register writes,
    /// sovereign dispatch becomes possible through coralReef + coral-driver.
    ///
    /// When this workaround is active, the device requires nvPmu init before
    /// any compute dispatch via the sovereign path.
    VoltaNoPmuFirmware,
}

/// Detect which workarounds apply for the given driver/arch combination.
pub(crate) fn detect_workarounds(driver: DriverKind, arch: GpuArch) -> Vec<Workaround> {
    let mut w = Vec::new();
    if driver == DriverKind::Nvk {
        w.push(Workaround::NvkExpF64Crash);
        w.push(Workaround::NvkLogF64Crash);
        w.push(Workaround::NvkLargeBufferLimit);
        w.push(Workaround::NvkSinCosF64Imprecise);
    }
    if driver == DriverKind::NvidiaProprietary && arch == GpuArch::Ada {
        w.push(Workaround::NvvmAdaF64Transcendentals);
    }
    // naga WGSL->SPIR-V codegen zeroes DF64 transcendentals on all Vulkan
    // backends. Root cause is naga, not the driver JIT (hotSpring Exp 055).
    w.push(Workaround::Df64SpirVPoisoning);
    // Volta desktop GPUs lack PMU firmware — nouveau cannot create compute
    // channels without it. nvPmu (software PMU) is required for sovereign
    // dispatch on these devices.
    if driver == DriverKind::Nvk && arch == GpuArch::Volta {
        w.push(Workaround::VoltaNoPmuFirmware);
    }
    w
}
