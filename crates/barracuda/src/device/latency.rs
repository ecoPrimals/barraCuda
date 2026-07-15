// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU instruction latency models for WGSL ILP scheduling.
//!
//! ## Background
//!
//! GPU shader compilers (NAK, ACO, PTXAS) perform instruction scheduling to
//! hide pipeline latencies. A **scoreboard** stall occurs when the next
//! instruction reads a register whose producing instruction has not yet
//! completed. Hiding that stall requires placing *independent* instructions
//! in the latency window — a technique called **instruction-level parallelism**
//! (ILP).
//!
//! `BarraCuda` expresses ILP at the WGSL source level (via `@ilp_region`
//! annotations and the Phase 3 `WgslDependencyGraph` reorderer). This module
//! provides the **latency numbers** that guide that reordering.
//!
//! ## Sources
//!
//! - SM70 (Volta): arXiv:1804.06826 — "Dissecting the NVIDIA Volta GPU"
//! - SM75/80/86/89: NAK `sm7x_instr_latencies.rs` (same DFMA=8cy pipeline)
//! - RDNA2: AMD RDNA2 ISA guide, empirical measurement via `bench_f64_builtins`
//! - Conservative: maximum observed across all families (safe for unknowns)
//!
//! ## Usage
//!
//! ```rust
//! use barracuda::device::latency::{LatencyModel, WgslOpClass};
//!
//! let model = LatencyModel::Sm70;
//! let ilp_window = model.raw_latency(WgslOpClass::F64Fma);
//! // ilp_window == 8 → place 8 independent ops in the Jacobi rotation kernel
//! ```

/// Classification of WGSL operations by pipeline latency.
///
/// Latency values represent the **read-after-write** (RAW) stall in cycles
/// between the producing instruction and the first dependent read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WgslOpClass {
    /// FP64 fused multiply-add: `a * b + c`.
    /// SM70 DFMA: 8 cycles. RDNA2 VFMA64: ~4 cycles.
    F64Fma,

    /// FP64 multiply then add (two separate instructions).
    /// Approximately 2× the single FMA latency.
    F64MulAdd,

    /// FP64 transcendentals (`exp_f64`, `log_f64`, `sqrt` in software).
    /// ~20 cycles — these are multi-instruction sequences.
    F64Transcend,

    /// FP32 fused multiply-add.
    /// SM70 FFMA: 4 cycles. RDNA2: ~4 cycles.
    F32Fma,

    /// Integer arithmetic (IADD, IMAD).
    /// 2–6 cycles. Useful as ILP filler alongside FP64.
    I32Arith,

    /// Shared memory load.
    /// ~20–30 cycles to return data (bank conflicts aside).
    SharedMem,

    /// Global memory load.
    /// 200–800 cycles (highly variable with L2 hit rate).
    GlobalMem,
}

/// GPU instruction latency model.
///
/// Enum dispatch over a closed set of GPU microarchitectures. Each variant
/// provides per-operation cycle counts that the WGSL optimizer uses to
/// schedule independent instructions into latency windows.
///
/// 4 of 5 variants are zero-sized (no data); `Measured` carries 3 empirical
/// cycle counts from `bench_f64_builtins`. The enum is `Copy` when all
/// variants are data-free; with `Measured` it is `Clone` only.
#[derive(Debug, Clone)]
pub enum LatencyModel {
    /// SM70 (Volta) — 8-cycle DFMA pipeline.
    /// Applies to SM75 (Turing), SM80/86 (Ampere), SM89 (Ada), SM100+ (Blackwell).
    Sm70,

    /// RDNA2 — 4-cycle VFMA64 pipeline.
    /// Applies to RDNA3 and CDNA2 as well.
    Rdna2,

    /// Conservative — maximum observed latency across all families.
    /// Safe for unknown GPUs; over-schedules but never under-schedules.
    Conservative,

    /// Apple M-series — software-emulated f64 (~16cy FMA).
    AppleM,

    /// Measured model from `bench_f64_builtins` output.
    /// Empirical cycle counts for the actual target GPU.
    Measured {
        /// FP64 FMA latency in cycles.
        dfma_cycles: u32,
        /// FP32 FMA latency in cycles.
        ffma_cycles: u32,
        /// Shared memory load latency in cycles.
        smem_cycles: u32,
    },
}

impl LatencyModel {
    /// Create a measured model from benchmark results.
    #[must_use]
    pub const fn measured(dfma_cycles: u32, ffma_cycles: u32, smem_cycles: u32) -> Self {
        Self::Measured {
            dfma_cycles,
            ffma_cycles,
            smem_cycles,
        }
    }

    /// Cycles between a write and the first valid dependent read (RAW latency).
    ///
    /// Place this many independent instructions between a producing `let`
    /// binding and its first use to avoid a scoreboard stall.
    #[must_use]
    pub fn raw_latency(&self, op: WgslOpClass) -> u32 {
        match self {
            Self::Sm70 => match op {
                WgslOpClass::F64Fma => 8,
                WgslOpClass::F64MulAdd => 16,
                WgslOpClass::F64Transcend => 20,
                WgslOpClass::F32Fma => 4,
                WgslOpClass::I32Arith => 6,
                WgslOpClass::SharedMem => 25,
                WgslOpClass::GlobalMem => 300,
            },
            Self::Rdna2 => match op {
                WgslOpClass::F64Fma => 4,
                WgslOpClass::F64MulAdd => 8,
                WgslOpClass::F64Transcend => 16,
                WgslOpClass::F32Fma => 4,
                WgslOpClass::I32Arith => 4,
                WgslOpClass::SharedMem => 20,
                WgslOpClass::GlobalMem => 200,
            },
            Self::Conservative => match op {
                WgslOpClass::F64Fma => 8,
                WgslOpClass::F64MulAdd => 16,
                WgslOpClass::F64Transcend => 20,
                WgslOpClass::F32Fma => 4,
                WgslOpClass::I32Arith => 6,
                WgslOpClass::SharedMem => 30,
                WgslOpClass::GlobalMem => 800,
            },
            Self::AppleM => match op {
                WgslOpClass::F64Fma => 16,
                WgslOpClass::F64MulAdd => 32,
                WgslOpClass::F64Transcend => 40,
                WgslOpClass::F32Fma => 4,
                WgslOpClass::I32Arith => 4,
                WgslOpClass::SharedMem => 20,
                WgslOpClass::GlobalMem => 250,
            },
            Self::Measured {
                dfma_cycles,
                ffma_cycles,
                smem_cycles,
            } => match op {
                WgslOpClass::F64Fma => *dfma_cycles,
                WgslOpClass::F64MulAdd => dfma_cycles * 2,
                WgslOpClass::F64Transcend => dfma_cycles + 12,
                WgslOpClass::F32Fma => *ffma_cycles,
                WgslOpClass::I32Arith => Self::Conservative.raw_latency(op),
                WgslOpClass::SharedMem => *smem_cycles,
                WgslOpClass::GlobalMem => Self::Conservative.raw_latency(op),
            },
        }
    }

    /// Cycles between a read and the next overwrite of the same register (WAR).
    ///
    /// Typically 0 on register-renamed pipelines; relevant on simpler cores.
    #[must_use]
    pub fn war_latency(&self, _op: WgslOpClass) -> u32 {
        0
    }

    /// Whether this operation goes through a scoreboard (true) or a fixed
    /// pipeline with a known latency slot (false).
    #[must_use]
    pub fn needs_scoreboard(&self, op: WgslOpClass) -> bool {
        match self {
            Self::Conservative | Self::Measured { .. } => true,
            Self::Sm70 | Self::Rdna2 | Self::AppleM => matches!(
                op,
                WgslOpClass::F64Fma
                    | WgslOpClass::F64MulAdd
                    | WgslOpClass::F64Transcend
                    | WgslOpClass::SharedMem
                    | WgslOpClass::GlobalMem
            ),
        }
    }

    /// Recommended ILP fill width: how many independent FP64 FMAs to
    /// interleave before issuing the dependent instruction.
    #[must_use]
    pub fn f64_ilp_fill_width(&self) -> u32 {
        self.raw_latency(WgslOpClass::F64Fma)
    }
}

/// Select the appropriate `LatencyModel` from a `GpuArch`.
#[must_use]
pub fn model_for_arch(arch: super::driver_profile::GpuArch) -> LatencyModel {
    use super::driver_profile::GpuArch;
    match arch {
        GpuArch::Volta | GpuArch::Turing | GpuArch::Ampere | GpuArch::Ada | GpuArch::Blackwell => {
            LatencyModel::Sm70
        }
        GpuArch::Rdna2 | GpuArch::Rdna3 | GpuArch::Cdna2 => LatencyModel::Rdna2,
        GpuArch::IntelArc => LatencyModel::Conservative,
        GpuArch::AppleM => LatencyModel::AppleM,
        GpuArch::Software | GpuArch::Unknown => LatencyModel::Conservative,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm70_dfma_latency() {
        let m = LatencyModel::Sm70;
        assert_eq!(m.raw_latency(WgslOpClass::F64Fma), 8);
        assert_eq!(m.raw_latency(WgslOpClass::F32Fma), 4);
        assert_eq!(m.f64_ilp_fill_width(), 8);
    }

    #[test]
    fn test_rdna2_dfma_latency() {
        let m = LatencyModel::Rdna2;
        assert_eq!(m.raw_latency(WgslOpClass::F64Fma), 4);
        assert_eq!(m.raw_latency(WgslOpClass::F32Fma), 4);
        assert_eq!(m.f64_ilp_fill_width(), 4);
    }

    #[test]
    fn test_conservative_is_max() {
        let conservative = LatencyModel::Conservative;
        let sm70 = LatencyModel::Sm70;
        let rdna2 = LatencyModel::Rdna2;
        for op in [
            WgslOpClass::F64Fma,
            WgslOpClass::F32Fma,
            WgslOpClass::I32Arith,
            WgslOpClass::SharedMem,
        ] {
            assert!(
                conservative.raw_latency(op) >= sm70.raw_latency(op),
                "Conservative should be >= SM70 for {op:?}"
            );
            assert!(
                conservative.raw_latency(op) >= rdna2.raw_latency(op),
                "Conservative should be >= RDNA2 for {op:?}"
            );
        }
    }

    #[test]
    fn test_measured_model() {
        let m = LatencyModel::measured(8, 4, 25);
        assert_eq!(m.raw_latency(WgslOpClass::F64Fma), 8);
        assert_eq!(m.raw_latency(WgslOpClass::F32Fma), 4);
        assert_eq!(m.raw_latency(WgslOpClass::F64MulAdd), 16);
        assert_eq!(m.raw_latency(WgslOpClass::SharedMem), 25);
    }

    #[test]
    fn test_model_for_arch() {
        use super::super::driver_profile::GpuArch;
        let model = model_for_arch(GpuArch::Volta);
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 8);
        let model = model_for_arch(GpuArch::Rdna2);
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 4);
        let model = model_for_arch(GpuArch::Unknown);
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 8);
    }

    #[test]
    fn test_scoreboard_classification() {
        let m = LatencyModel::Sm70;
        assert!(m.needs_scoreboard(WgslOpClass::F64Fma));
        assert!(m.needs_scoreboard(WgslOpClass::GlobalMem));
        assert!(!m.needs_scoreboard(WgslOpClass::F32Fma));
        assert!(!m.needs_scoreboard(WgslOpClass::I32Arith));
    }

    #[test]
    fn test_war_latency_zero_on_register_renamed_gpus() {
        for op in [WgslOpClass::F64Fma, WgslOpClass::SharedMem] {
            assert_eq!(LatencyModel::Sm70.war_latency(op), 0);
            assert_eq!(LatencyModel::Rdna2.war_latency(op), 0);
        }
    }
}
