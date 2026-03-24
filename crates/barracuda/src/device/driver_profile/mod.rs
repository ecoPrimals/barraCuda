// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU driver identity types — enums for driver, compiler, architecture, and strategy.
//!
//! These types answer **"who is driving the hardware?"** and feed into
//! [`super::capabilities::DeviceCapabilities`] for workgroup sizing,
//! precision routing, and FP64 strategy selection.
//!
//! The former `GpuDriverProfile` struct was removed in v0.3.8 (Sprint 18).
//! All consumers migrated to `DeviceCapabilities` in Sprint 14.

mod architectures;
mod workarounds;

pub use architectures::GpuArch;
pub use workarounds::Workaround;

// ── Driver identity ───────────────────────────────────────────────────────────

/// GPU driver/compiler identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DriverKind {
    /// NVIDIA proprietary (NVVM/PTXAS)
    NvidiaProprietary,
    /// Mesa NVK (open-source NVIDIA Vulkan)
    Nvk,
    /// Mesa RADV (AMD Vulkan)
    Radv,
    /// Intel ANV
    Intel,
    /// Software rasterizer
    Software,
    /// Unknown driver
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
    /// Unknown compiler
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
    WarpPacked {
        /// Workgroup size (typically 32)
        wg_size: u32,
    },
    /// Wave-packed: 64 independent matrices per workgroup (AMD)
    WavePacked {
        /// Wavefront size (typically 64)
        wave_size: u32,
    },
    /// Standard: one matrix per workgroup
    Standard,
}

// ── FP64 execution strategy ──────────────────────────────────────────────────

/// Hardware-adaptive FP64 execution strategy (physics domain core-streaming discovery).
///
/// On compute-class GPUs (Titan V, V100, MI250) with 1:2 FP64:FP32 hardware,
/// native `f64` is fastest. On consumer GPUs (1:64 ratio), routing bulk math
/// through double-float f32-pair (`Df64`) on the massive FP32 core array
/// delivers ~10x the effective throughput, reserving native `f64` for
/// precision-critical reductions and convergence tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp64Strategy {
    /// Sovereign compilation via coralReef: WGSL → native GPU binary via IPC.
    /// Bypasses naga/SPIR-V entirely — resolves f64 shared-memory failures.
    /// Requires coralReef primal + coralDriver for dispatch; falls back
    /// through the chain: Sovereign → Native → Hybrid → f32.
    Sovereign,
    /// Use native f64 everywhere (Titan V, V100, A100, MI250X — 1:2 hardware).
    Native,
    /// DF64 (f32-pair, ~14 digits) for bulk math, native f64 for reductions.
    Hybrid,
    /// Run both DF64 and native f64 concurrently, cross-validate results.
    /// Useful for validation harnesses and precision-sensitive pipelines.
    Concurrent,
}

// ── Precision routing (toadStool S128 integration) ──────────────────────────

/// Precision routing advice from toadStool S128.
///
/// Captures the three-axis reality of GPU f64: some hardware has native f64
/// everywhere, some has native f64 but broken shared-memory f64 reductions
/// (naga/SPIR-V emit zeros), some can only do DF64, and some are f32-only.
///
/// Use this to route workloads to the correct precision path at dispatch time,
/// especially for reductions that rely on `var<workgroup>` f64 accumulators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionRoutingAdvice {
    /// Native f64 everywhere — compute and shared-memory reductions.
    F64Native,
    /// Native f64 compute works, but shared-memory f64 reductions return zeros.
    /// Route reductions through DF64 or scalar f64 (no workgroup accumulator).
    F64NativeNoSharedMem,
    /// No reliable native f64; use DF64 (f32-pair) for all f64-class work.
    Df64Only,
    /// No f64 support at all; fall back to f32.
    F32Only,
}
