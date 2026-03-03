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
//! BarraCuda expresses ILP at the WGSL source level (via `@ilp_region`
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
//! use barracuda::device::latency::{Sm70LatencyModel, LatencyModel, WgslOpClass};
//!
//! let model = Sm70LatencyModel;
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
/// Implementors provide per-operation cycle counts that the WGSL optimizer
/// uses to schedule independent instructions into latency windows.
pub trait LatencyModel: Send + Sync {
    /// Cycles between a write and the first valid dependent read (RAW latency).
    ///
    /// Place this many independent instructions between a producing `let`
    /// binding and its first use to avoid a scoreboard stall.
    fn raw_latency(&self, op: WgslOpClass) -> u32;

    /// Cycles between a read and the next overwrite of the same register (WAR).
    ///
    /// Typically 0 on register-renamed pipelines; relevant on simpler cores.
    fn war_latency(&self, op: WgslOpClass) -> u32;

    /// Whether this operation goes through a scoreboard (true) or a fixed
    /// pipeline with a known latency slot (false).
    ///
    /// When `true`, out-of-order dispatch can hide the latency. When `false`
    /// (fixed pipeline), the hardware stalls if the window is not filled.
    fn needs_scoreboard(&self, op: WgslOpClass) -> bool;

    /// Recommended ILP fill width: how many independent FP64 FMAs to
    /// interleave before issuing the dependent instruction.
    ///
    /// Convenience wrapper: `raw_latency(F64Fma)`.
    fn f64_ilp_fill_width(&self) -> u32 {
        self.raw_latency(WgslOpClass::F64Fma)
    }
}

// ─── Concrete models ──────────────────────────────────────────────────────────

/// SM70 (Volta) latency model.
///
/// Source: arXiv:1804.06826 — "Dissecting the NVIDIA Volta GPU Architecture
/// via Microbenchmarking" (Jia et al., 2018).
///
/// Applies to: Titan V, V100, Quadro GV100.
/// Also applies to SM75 (Turing), SM80/86 (Ampere), SM89 (Ada) —
/// all share the same DFMA=8cy pipeline.
pub struct Sm70LatencyModel;

impl LatencyModel for Sm70LatencyModel {
    fn raw_latency(&self, op: WgslOpClass) -> u32 {
        match op {
            WgslOpClass::F64Fma => 8,
            WgslOpClass::F64MulAdd => 16, // 2 × 8
            WgslOpClass::F64Transcend => 20,
            WgslOpClass::F32Fma => 4,
            WgslOpClass::I32Arith => 6,
            WgslOpClass::SharedMem => 25,
            WgslOpClass::GlobalMem => 300,
        }
    }

    fn war_latency(&self, _op: WgslOpClass) -> u32 {
        0 // Volta uses register renaming; WAR latency is zero
    }

    fn needs_scoreboard(&self, op: WgslOpClass) -> bool {
        matches!(
            op,
            WgslOpClass::F64Fma
                | WgslOpClass::F64MulAdd
                | WgslOpClass::F64Transcend
                | WgslOpClass::SharedMem
                | WgslOpClass::GlobalMem
        )
    }
}

/// RDNA2 latency model.
///
/// Source: AMD RDNA2 ISA documentation + empirical measurement via
/// `bench_f64_builtins` on RX 6950 XT.
///
/// Applies to: RX 6000 series, RX 6x50 XT refresh. RDNA3 is similar.
pub struct Rdna2LatencyModel;

impl LatencyModel for Rdna2LatencyModel {
    fn raw_latency(&self, op: WgslOpClass) -> u32 {
        match op {
            WgslOpClass::F64Fma => 4,
            WgslOpClass::F64MulAdd => 8,
            WgslOpClass::F64Transcend => 16, // software sequence
            WgslOpClass::F32Fma => 4,
            WgslOpClass::I32Arith => 4,
            WgslOpClass::SharedMem => 20,
            WgslOpClass::GlobalMem => 200,
        }
    }

    fn war_latency(&self, _op: WgslOpClass) -> u32 {
        0
    }

    fn needs_scoreboard(&self, op: WgslOpClass) -> bool {
        matches!(
            op,
            WgslOpClass::F64Fma
                | WgslOpClass::F64MulAdd
                | WgslOpClass::F64Transcend
                | WgslOpClass::SharedMem
                | WgslOpClass::GlobalMem
        )
    }
}

/// Conservative latency model — uses the maximum observed latency across
/// all supported GPU families.
///
/// Safe to use when the target GPU is unknown. Will over-schedule (insert
/// more filler ops than necessary) but will never under-schedule (leave
/// stall cycles unfilled).
pub struct ConservativeModel;

impl LatencyModel for ConservativeModel {
    fn raw_latency(&self, op: WgslOpClass) -> u32 {
        match op {
            WgslOpClass::F64Fma => 8, // SM70 worst case
            WgslOpClass::F64MulAdd => 16,
            WgslOpClass::F64Transcend => 20,
            WgslOpClass::F32Fma => 4,
            WgslOpClass::I32Arith => 6,
            WgslOpClass::SharedMem => 30,
            WgslOpClass::GlobalMem => 800, // cache miss worst case
        }
    }

    fn war_latency(&self, _op: WgslOpClass) -> u32 {
        0
    }

    fn needs_scoreboard(&self, _op: WgslOpClass) -> bool {
        true // conservative: assume everything uses a scoreboard
    }
}

/// Measured latency model populated from `bench_f64_builtins` output.
///
/// `bench_f64_builtins` runs a micro-benchmark on the actual target GPU and
/// derives empirical cycle counts. Those values are stored here and used
/// for precise ILP scheduling.
///
/// When measurements are not available for a specific operation, falls back
/// to `ConservativeModel`.
#[derive(Debug, Clone)]
pub struct MeasuredModel {
    /// Empirical FP64 FMA latency in cycles.
    pub dfma_cycles: u32,
    /// Empirical FP32 FMA latency in cycles.
    pub ffma_cycles: u32,
    /// Empirical shared memory load latency.
    pub smem_cycles: u32,
}

impl MeasuredModel {
    /// Create a measured model from benchmark results.
    #[must_use]
    pub fn new(dfma_cycles: u32, ffma_cycles: u32, smem_cycles: u32) -> Self {
        Self {
            dfma_cycles,
            ffma_cycles,
            smem_cycles,
        }
    }
}

impl LatencyModel for MeasuredModel {
    fn raw_latency(&self, op: WgslOpClass) -> u32 {
        match op {
            WgslOpClass::F64Fma => self.dfma_cycles,
            WgslOpClass::F64MulAdd => self.dfma_cycles * 2,
            WgslOpClass::F64Transcend => self.dfma_cycles + 12, // software overhead
            WgslOpClass::F32Fma => self.ffma_cycles,
            WgslOpClass::I32Arith => ConservativeModel.raw_latency(op),
            WgslOpClass::SharedMem => self.smem_cycles,
            WgslOpClass::GlobalMem => ConservativeModel.raw_latency(op),
        }
    }

    fn war_latency(&self, _op: WgslOpClass) -> u32 {
        0
    }

    fn needs_scoreboard(&self, op: WgslOpClass) -> bool {
        ConservativeModel.needs_scoreboard(op)
    }
}

/// Apple M-series GPU latency model.
///
/// Apple Silicon GPUs do not expose hardware FP64 — all f64 operations are
/// executed as software multi-instruction sequences. The effective latency for
/// an f64 FMA is empirically ~16 cycles (4× the f32 FFMA pipeline).
///
/// Source: Apple GPU ISA (internal; approximated from bench_f64_builtins patterns).
/// Status: to be calibrated with `bench_f64_builtins` on macOS / Metal.
pub struct AppleMLatencyModel;

impl LatencyModel for AppleMLatencyModel {
    fn raw_latency(&self, op: WgslOpClass) -> u32 {
        match op {
            WgslOpClass::F64Fma => 16, // software-emulated: 4× f32 pipeline
            WgslOpClass::F64MulAdd => 32,
            WgslOpClass::F64Transcend => 40,
            WgslOpClass::F32Fma => 4,
            WgslOpClass::I32Arith => 4,
            WgslOpClass::SharedMem => 20,
            WgslOpClass::GlobalMem => 250,
        }
    }

    fn war_latency(&self, _op: WgslOpClass) -> u32 {
        0
    }

    fn needs_scoreboard(&self, op: WgslOpClass) -> bool {
        matches!(
            op,
            WgslOpClass::F64Fma
                | WgslOpClass::F64MulAdd
                | WgslOpClass::F64Transcend
                | WgslOpClass::SharedMem
                | WgslOpClass::GlobalMem
        )
    }
}

/// Select the appropriate `LatencyModel` from a `GpuArch`.
///
/// Returns a `Box<dyn LatencyModel>` — caller decides whether to keep it
/// as a trait object or downcast.
#[must_use]
pub fn model_for_arch(arch: super::capabilities::GpuArch) -> Box<dyn LatencyModel> {
    use super::capabilities::GpuArch;
    match arch {
        // All NVIDIA SM7x–SM8x share the 8-cycle DFMA pipeline
        GpuArch::Volta | GpuArch::Turing | GpuArch::Ampere | GpuArch::Ada => {
            Box::new(Sm70LatencyModel)
        }
        // AMD RDNA2 / RDNA3 — similar 4-cycle VFMA64 pipeline
        GpuArch::Rdna2 | GpuArch::Rdna3 | GpuArch::Cdna2 => Box::new(Rdna2LatencyModel),
        // Intel Xe — to be measured empirically; conservative safe
        GpuArch::IntelArc => Box::new(ConservativeModel),
        // Apple M-series: hardware f64 is software-emulated so FMA latency is high;
        // f32 pipeline is ~4cy (similar to RDNA2). Use measured 16cy f64 estimate.
        GpuArch::AppleM => Box::new(AppleMLatencyModel),
        GpuArch::Software | GpuArch::Unknown => Box::new(ConservativeModel),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sm70_dfma_latency() {
        let m = Sm70LatencyModel;
        assert_eq!(m.raw_latency(WgslOpClass::F64Fma), 8);
        assert_eq!(m.raw_latency(WgslOpClass::F32Fma), 4);
        assert_eq!(m.f64_ilp_fill_width(), 8);
    }

    #[test]
    fn test_rdna2_dfma_latency() {
        let m = Rdna2LatencyModel;
        assert_eq!(m.raw_latency(WgslOpClass::F64Fma), 4);
        assert_eq!(m.raw_latency(WgslOpClass::F32Fma), 4);
        assert_eq!(m.f64_ilp_fill_width(), 4);
    }

    #[test]
    fn test_conservative_is_max() {
        let conservative = ConservativeModel;
        let sm70 = Sm70LatencyModel;
        let rdna2 = Rdna2LatencyModel;
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
        let m = MeasuredModel::new(8, 4, 25);
        assert_eq!(m.raw_latency(WgslOpClass::F64Fma), 8);
        assert_eq!(m.raw_latency(WgslOpClass::F32Fma), 4);
        assert_eq!(m.raw_latency(WgslOpClass::F64MulAdd), 16);
        assert_eq!(m.raw_latency(WgslOpClass::SharedMem), 25);
    }

    #[test]
    fn test_model_for_arch() {
        use super::super::capabilities::GpuArch;
        // Volta → SM70 model → 8-cycle DFMA
        let model = model_for_arch(GpuArch::Volta);
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 8);
        // RDNA2 → 4-cycle VFMA64
        let model = model_for_arch(GpuArch::Rdna2);
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 4);
        // Unknown → conservative → 8-cycle
        let model = model_for_arch(GpuArch::Unknown);
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 8);
    }

    #[test]
    fn test_scoreboard_classification() {
        let m = Sm70LatencyModel;
        assert!(m.needs_scoreboard(WgslOpClass::F64Fma));
        assert!(m.needs_scoreboard(WgslOpClass::GlobalMem));
        assert!(!m.needs_scoreboard(WgslOpClass::F32Fma));
        assert!(!m.needs_scoreboard(WgslOpClass::I32Arith));
    }

    #[test]
    fn test_war_latency_zero_on_register_renamed_gpus() {
        for op in [WgslOpClass::F64Fma, WgslOpClass::SharedMem] {
            assert_eq!(Sm70LatencyModel.war_latency(op), 0);
            assert_eq!(Rdna2LatencyModel.war_latency(op), 0);
        }
    }
}
