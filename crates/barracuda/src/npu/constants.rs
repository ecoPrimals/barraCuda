// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU hardware constants validated by metalForge probing
//!
//! These values are derived from systematic hardware testing on a physical
//! BrainChip AKD1000, overturning several SDK assumptions. See
//! `ecoPrimals/hotSpring/metalForge/npu/akida/BEYOND_SDK.md` for methodology.
//!
//! All tolerances represent worst-case measurements from the hardware probe
//! suite (`deep_probe.py`, 8 test suites, all passing on AKD1000).

/// FC chain depth overhead: 7 extra FC layers vs 1 FC layer.
/// Measured at 6.7% on AKD1000 — all FC layers merge into a single
/// hardware sequence via intra-mesh SkipDMA.
pub const FC_DEPTH_OVERHEAD_MAX: f64 = 0.30;

/// Batch inference speedup floor: batch=8 vs batch=1.
/// Measured at 2.35x on AKD1000 (427 us/sample at batch=8).
/// PCIe round-trip amortization is the mechanism.
pub const BATCH_SPEEDUP_MIN: f64 = 1.5;

/// Multi-output overhead: 10 outputs vs 1 output.
/// Measured at 4.5% on AKD1000.
pub const MULTI_OUTPUT_OVERHEAD_MAX: f64 = 0.30;

/// Weight mutation linearity tolerance: set_variable(weights * k)
/// should produce output * k within this error bound.
/// Measured at 0.0000 on AKD1000 (exact integer linearity).
pub const WEIGHT_MUTATION_LINEARITY: f64 = 0.01;

/// Optimal batch size for PCIe amortization.
/// Beyond 16, SRAM contention degrades throughput on AKD1000.
pub const OPTIMAL_BATCH_SIZE: u32 = 8;

/// Maximum tested input channel count that maps to hardware.
/// SDK documents 1 or 3; hardware accepts any count (tested 1-64).
pub const MAX_TESTED_INPUT_CHANNELS: u32 = 64;

/// Maximum tested FC layer width that maps to hardware.
/// SDK documents "hundreds"; tested to 8192+ on AKD1000.
pub const MAX_TESTED_FC_WIDTH: u32 = 8192;

/// Weight update overhead in microseconds.
/// `set_variable()` + forward vs forward alone.
pub const WEIGHT_UPDATE_OVERHEAD_US: f64 = 14_000.0;

/// Economy clock mode speed penalty (fraction slower than Performance).
/// Measured at 19% on AKD1000.
pub const ECONOMY_CLOCK_SPEED_PENALTY: f64 = 0.19;

/// Economy clock mode power savings (fraction less than Performance).
/// Measured at 18% on AKD1000.
pub const ECONOMY_CLOCK_POWER_SAVINGS: f64 = 0.18;

/// Quantization error budgets for NPU deployment.
/// These define acceptable error vs f64 reference for each precision level.
pub mod quantization {
    /// f32 max error vs f64 — acceptable for all physics workloads.
    pub const F32_MAX_ERROR: f64 = 0.00001;

    /// int8 max error vs f64 — within MD statistical uncertainty.
    pub const INT8_MAX_ERROR: f64 = 0.05;

    /// int4 max error vs f64 — marginal, use for screening only.
    pub const INT4_MAX_ERROR: f64 = 0.30;

    /// int4 with 4-bit activations — too lossy for physics.
    pub const INT4_ACT4_MAX_ERROR: f64 = 0.50;
}

// Compile-time validation: constants stay within physical bounds.
const _: () = {
    assert!(FC_DEPTH_OVERHEAD_MAX > 0.0);
    assert!(FC_DEPTH_OVERHEAD_MAX < 1.0);
    assert!(BATCH_SPEEDUP_MIN > 1.0);
    assert!(MULTI_OUTPUT_OVERHEAD_MAX > 0.0);
    assert!(MULTI_OUTPUT_OVERHEAD_MAX < 1.0);
    assert!(WEIGHT_MUTATION_LINEARITY > 0.0);
    assert!(WEIGHT_MUTATION_LINEARITY < 0.1);
    assert!(OPTIMAL_BATCH_SIZE > 0);
    assert!(OPTIMAL_BATCH_SIZE <= 32);
    assert!(ECONOMY_CLOCK_SPEED_PENALTY > 0.0);
    assert!(ECONOMY_CLOCK_SPEED_PENALTY < 0.5);
    assert!(ECONOMY_CLOCK_POWER_SAVINGS > 0.0);
    assert!(ECONOMY_CLOCK_POWER_SAVINGS < 0.5);
};

const _: () = {
    assert!(quantization::F32_MAX_ERROR < quantization::INT8_MAX_ERROR);
    assert!(quantization::INT8_MAX_ERROR < quantization::INT4_MAX_ERROR);
    assert!(quantization::INT4_MAX_ERROR < quantization::INT4_ACT4_MAX_ERROR);
};
