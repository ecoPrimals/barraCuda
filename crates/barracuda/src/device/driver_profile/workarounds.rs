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
    /// NVIDIA proprietary NVVM device poisoning: DF64 or `F64Precise` shaders
    /// containing f64 transcendentals (exp, log, pow) cause an unrecoverable
    /// NVVM compilation failure that permanently invalidates the wgpu device.
    /// All subsequent buffer creation, dispatch, and readback panic with
    /// "Buffer is invalid". Only recovery is process restart.
    ///
    /// Discovered by hotSpring v0.6.25 on RTX 3090 (Ampere, SM86). Affects
    /// all NVIDIA proprietary driver architectures — the NVVM compiler cannot
    /// handle f64 transcendentals in DF64 (f32-pair rewrite) or `F64Precise`
    /// (no-FMA) compilation modes. NVK (Mesa) is unaffected.
    NvvmDf64TranscendentalPoisoning,
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
    if driver == DriverKind::NvidiaProprietary {
        if arch == GpuArch::Ada {
            w.push(Workaround::NvvmAdaF64Transcendentals);
        }
        w.push(Workaround::NvvmDf64TranscendentalPoisoning);
    }
    w
}
