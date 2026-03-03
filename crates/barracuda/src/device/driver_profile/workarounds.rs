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
    /// Conservative limit of 1.2 GB total. File upstream bug in drm_gpuvm.
    NvkLargeBufferLimit,
    /// NVK has broken or imprecise sin/cos/tan for f64. Use Taylor series
    /// approximations (sin_f64_safe, cos_f64_safe) instead of native or polyfill.
    NvkSinCosF64Imprecise,
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
    w
}
