// SPDX-License-Identifier: AGPL-3.0-or-later

//! Precision-tier tolerances from `PRECISION_TIERS_SPECIFICATION.md` §7.
//!
//! Each precision tier maps to an absolute and relative tolerance reflecting
//! the tier's mantissa width and numerical characteristics. The
//! `PrecisionBrain` uses these to validate results and escalate tiers.

use super::Tolerance;

/// DF128 (~104-bit mantissa, ~31 decimal digits).
pub const PRECISION_DF128: Tolerance = Tolerance {
    name: "precision::DF128",
    abs_tol: 1e-28,
    rel_tol: 1e-26,
    justification: "~104-bit mantissa from Dekker double-double on f64; ~31 digits",
};

/// QF128 (~96-bit mantissa, ~28 decimal digits).
pub const PRECISION_QF128: Tolerance = Tolerance {
    name: "precision::QF128",
    abs_tol: 1e-24,
    rel_tol: 1e-22,
    justification: "~96-bit mantissa from Bailey quad-double on f32; 4× Dekker error",
};

/// `F64Precise` (52-bit mantissa, FMA-free, gold standard).
pub const PRECISION_F64_PRECISE: Tolerance = Tolerance {
    name: "precision::F64Precise",
    abs_tol: 1e-14,
    rel_tol: 1e-12,
    justification: "native f64 without FMA fusion; cancellation-safe reference",
};

/// F64 (52-bit mantissa, FMA may introduce 1 ULP).
pub const PRECISION_F64: Tolerance = Tolerance {
    name: "precision::F64",
    abs_tol: 1e-12,
    rel_tol: 1e-10,
    justification: "native f64 with FMA; 1 ULP rounding from multiply-add fusion",
};

/// DF64 (~48-bit mantissa, Dekker error accumulation).
pub const PRECISION_DF64: Tolerance = Tolerance {
    name: "precision::DF64",
    abs_tol: 1e-10,
    rel_tol: 1e-8,
    justification: "f32-pair double-float; Dekker error accumulates across operations",
};

/// F32 (23-bit mantissa, standard GPU baseline).
pub const PRECISION_F32: Tolerance = Tolerance {
    name: "precision::F32",
    abs_tol: 1e-5,
    rel_tol: 1e-4,
    justification: "23-bit mantissa; ~7.2 decimal digits",
};

/// TF32 (10-bit mantissa, NVIDIA tensor core internal format).
pub const PRECISION_TF32: Tolerance = Tolerance {
    name: "precision::TF32",
    abs_tol: 1e-2,
    rel_tol: 5e-2,
    justification: "10-bit mantissa (same as f16); f32 exponent range preserved",
};

/// F16 (10-bit mantissa, IEEE half precision).
pub const PRECISION_F16: Tolerance = Tolerance {
    name: "precision::F16",
    abs_tol: 1e-2,
    rel_tol: 5e-2,
    justification: "10-bit mantissa; ~3.3 decimal digits; range ±65504",
};

/// BF16 (7-bit mantissa, bfloat16).
pub const PRECISION_BF16: Tolerance = Tolerance {
    name: "precision::BF16",
    abs_tol: 5e-2,
    rel_tol: 1e-1,
    justification: "7-bit mantissa; ~2.1 decimal digits; f32 exponent range",
};

/// FP8 E4M3 (3-bit mantissa, inference weights).
pub const PRECISION_FP8_E4M3: Tolerance = Tolerance {
    name: "precision::FP8_E4M3",
    abs_tol: 0.5,
    rel_tol: 0.25,
    justification: "3-bit mantissa; only 8 mantissa values; range ±448",
};

/// FP8 E5M2 (2-bit mantissa, gradient communication).
pub const PRECISION_FP8_E5M2: Tolerance = Tolerance {
    name: "precision::FP8_E5M2",
    abs_tol: 1.0,
    rel_tol: 0.5,
    justification: "2-bit mantissa; 4 mantissa values; range ±57344",
};

/// Q8 (8-bit block quantized): tolerance depends on scale factor.
pub const PRECISION_Q8: Tolerance = Tolerance {
    name: "precision::Q8",
    abs_tol: 0.5,
    rel_tol: 0.1,
    justification: "block quantized; dequant error = scale × 0.5 per element",
};

/// Q4 (4-bit block quantized): tolerance depends on scale factor.
pub const PRECISION_Q4: Tolerance = Tolerance {
    name: "precision::Q4",
    abs_tol: 1.0,
    rel_tol: 0.25,
    justification: "block quantized; 16 levels; dequant error = scale × 0.5",
};

/// Int2 / Ternary: exact integer arithmetic on 3 discrete values.
pub const PRECISION_INT2: Tolerance = Tolerance {
    name: "precision::INT2",
    abs_tol: 0.0,
    rel_tol: f64::EPSILON,
    justification: "3 discrete values {-1, 0, +1}; exact integer arithmetic",
};

/// Binary: exact 2-value arithmetic (XNOR+popcount).
pub const PRECISION_BINARY: Tolerance = Tolerance {
    name: "precision::Binary",
    abs_tol: 0.0,
    rel_tol: f64::EPSILON,
    justification: "2 values {0, 1}; XNOR+popcount; deterministic",
};

/// Look up the tolerance for a given precision tier.
///
/// Returns the spec-defined tolerance that reflects each tier's mantissa
/// width and numerical error characteristics. Use this to validate that
/// a tier is producing correct results and to decide when to escalate.
#[must_use]
pub fn for_precision_tier(
    tier: crate::device::precision_tier::PrecisionTier,
) -> &'static Tolerance {
    use crate::device::precision_tier::PrecisionTier;
    match tier {
        PrecisionTier::Binary => &PRECISION_BINARY,
        PrecisionTier::Int2 => &PRECISION_INT2,
        PrecisionTier::Quantized4 => &PRECISION_Q4,
        PrecisionTier::Quantized8 => &PRECISION_Q8,
        PrecisionTier::Fp8E5M2 => &PRECISION_FP8_E5M2,
        PrecisionTier::Fp8E4M3 => &PRECISION_FP8_E4M3,
        PrecisionTier::Bf16 => &PRECISION_BF16,
        PrecisionTier::F16 => &PRECISION_F16,
        PrecisionTier::Tf32 => &PRECISION_TF32,
        PrecisionTier::F32 => &PRECISION_F32,
        PrecisionTier::DF64 => &PRECISION_DF64,
        PrecisionTier::F64 => &PRECISION_F64,
        PrecisionTier::F64Precise => &PRECISION_F64_PRECISE,
        PrecisionTier::QF128 => &PRECISION_QF128,
        PrecisionTier::DF128 => &PRECISION_DF128,
    }
}
