// SPDX-License-Identifier: AGPL-3.0-or-later
//! Generic Precision Shader System
//!
//! Provides compile-time and runtime shader generation for any precision type.
//! ONE template → shaders for f16, f32, f64, and CPU implementations.

pub mod compiler;
pub mod cpu;
pub mod eps;
mod math_f64;
pub mod polyfill;

// Re-export public downcast/compiler API for external callers
pub use compiler::{
    downcast_f64_to_df64, downcast_f64_to_f32, downcast_f64_to_f32_with_transcendentals,
};

/// Hardware precision tiers for shader generation.
///
/// Math is written in f64-canonical WGSL — pure math, conceptually infinite
/// precision. The compilation pipeline then targets one of these hardware
/// tiers. Each tier maps to a coralReef compilation strategy.
///
/// | Tier | coralReef strategy | Mantissa | Notes |
/// |------|-------------------|----------|-------|
/// | Binary | `binary` | 1 bit | XNOR+popcount, u32 packed |
/// | Int2 | `int2` | 2 bits | Ternary {-1,0,+1}, u32 packed |
/// | Q4 | `q4_block` | 4 bits | Block-quantized Q4_0 |
/// | Q8 | `q8_block` | 8 bits | Block-quantized Q8_0 |
/// | Fp8E5M2 | `fp8_e5m2` | 2-bit mant | Gradient comm, u32 packed |
/// | Fp8E4M3 | `fp8_e4m3` | 3-bit mant | Inference, u32 packed |
/// | Bf16 | `bf16_emulated` | 7 bits | bfloat16, u32 bit-manip |
/// | F16 | `f16_fast` | 10 bits | IEEE half, native or emulated |
/// | F32 | `f32_only` | 24 bits | Universal baseline |
/// | Df64 | `double_float` | ~48 bits | f32-pair Dekker |
/// | F64 | `native` | 52 bits | Native f64 |
/// | Qf128 | `quad_float` | ~96 bits | Bailey quad-double on f32 |
/// | Df128 | `double_double_f64` | ~104 bits | Dekker double-double on f64 |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Precision {
    /// 1-bit binary: XNOR+popcount dot products, 32 values per u32.
    Binary,
    /// 2-bit ternary {-1, 0, +1}: 16 values per u32.
    Int2,
    /// 4-bit block quantized (`Q4_0`): 8 nibbles per u32 + f16 scale.
    Q4,
    /// 8-bit block quantized (`Q8_0`): 4 bytes per u32 + f16 scale.
    Q8,
    /// 8-bit float E5M2: wider range, 2-bit mantissa. 4 per u32.
    Fp8E5M2,
    /// 8-bit float E4M3: higher precision, 3-bit mantissa. 4 per u32.
    Fp8E4M3,
    /// 16-bit bfloat (Google Brain): f32 exponent range, 7-bit mantissa.
    /// Emulated via u32 bit manipulation. No wgpu feature required.
    Bf16,
    /// 16-bit float (half precision) — ML inference, screening, tensor core path.
    /// Requires `SHADER_F16` feature. Falls back to f32 pack/unpack emulation
    /// on hardware without native f16 support.
    F16,
    /// 32-bit float (single precision) — default, broadly supported.
    /// coralReef: `Fp64Strategy::F32Only`.
    F32,
    /// 64-bit float (double precision) — scientific computing, gold standard.
    /// coralReef: `Fp64Strategy::Native`.
    F64,
    /// Double-float f32-pair (~48-bit mantissa, ~14 decimal digits) —
    /// unleashes FP32 cores for f64-class work. 12–18× throughput vs native
    /// f64 on consumer GPUs. The "fp48" sweet spot.
    /// coralReef: `Fp64Strategy::DoubleFloat`.
    Df64,
    /// Quad-double on f32 (Bailey): ~96-bit mantissa from 4× f32 components.
    /// No f64 hardware required — universally available.
    /// coralReef: `Fp64Strategy::QuadFloat`.
    Qf128,
    /// Double-double on f64 (Dekker): ~104-bit mantissa from 2× f64 components.
    /// Requires `SHADER_F64`. Preferred over Qf128 on compute GPUs.
    /// coralReef: `Fp64Strategy::DoubleDoubleF64`.
    Df128,
}

impl Precision {
    /// WGSL scalar type name.
    #[must_use]
    pub fn scalar(&self) -> &'static str {
        match self {
            Self::Binary | Self::Int2 | Self::Q4 | Self::Q8 => "u32",
            Self::Fp8E5M2 | Self::Fp8E4M3 => "u32",
            Self::Bf16 => "u32",
            Self::F16 => "f16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::Df64 => "vec2<f32>",
            Self::Qf128 => "vec4<f32>",
            Self::Df128 => "vec2<f64>",
        }
    }

    /// WGSL vec2 type name (or scalar for types that lack native vec support).
    #[must_use]
    pub fn vec2(&self) -> &'static str {
        match self {
            Self::F16 => "vec2<f16>",
            Self::F32 => "vec2<f32>",
            Self::F64 => "f64",
            Self::Df64 => "vec2<f32>",
            Self::Qf128 => "vec4<f32>",
            Self::Df128 => "vec2<f64>",
            _ => "u32",
        }
    }

    /// WGSL vec4 type name (or scalar for types without vec support).
    #[must_use]
    pub fn vec4(&self) -> &'static str {
        match self {
            Self::F16 => "vec4<f16>",
            Self::F32 => "vec4<f32>",
            Self::F64 => "f64",
            Self::Df64 => "vec2<f32>",
            Self::Qf128 => "vec4<f32>",
            Self::Df128 => "vec2<f64>",
            _ => "u32",
        }
    }

    /// Whether this precision supports vectorized operations (vec4).
    #[must_use]
    pub fn has_vec4(&self) -> bool {
        matches!(self, Self::F16 | Self::F32)
    }

    /// Bytes per element.
    #[must_use]
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Binary | Self::Int2 | Self::Q4 | Self::Q8 => 1,
            Self::Fp8E5M2 | Self::Fp8E4M3 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 | Self::Df64 => 8,
            Self::Qf128 | Self::Df128 => 16,
        }
    }

    /// Required wgpu feature for this precision.
    #[must_use]
    pub fn required_feature(&self) -> Option<wgpu::Features> {
        match self {
            Self::F16 => Some(wgpu::Features::SHADER_F16),
            Self::F64 | Self::Df128 => Some(wgpu::Features::SHADER_F64),
            _ => None,
        }
    }

    /// Whether this is an f64-class precision (native f64 or df64 emulation).
    #[must_use]
    pub fn is_f64_class(&self) -> bool {
        matches!(self, Self::F64 | Self::Df64 | Self::Df128)
    }

    /// Whether this is a reduced-precision tier (below f32).
    #[must_use]
    pub fn is_reduced(&self) -> bool {
        matches!(
            self,
            Self::Binary
                | Self::Int2
                | Self::Q4
                | Self::Q8
                | Self::Fp8E5M2
                | Self::Fp8E4M3
                | Self::Bf16
                | Self::F16
        )
    }

    /// Whether this is an extended-precision tier (above f64).
    #[must_use]
    pub fn is_extended(&self) -> bool {
        matches!(self, Self::Qf128 | Self::Df128)
    }

    /// Whether this is a quantized integer format.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        matches!(self, Self::Binary | Self::Int2 | Self::Q4 | Self::Q8)
    }

    /// Generate the operation preamble for this precision.
    ///
    /// The preamble defines abstract operations (`op_add`, `op_mul`, etc.)
    /// whose implementation varies per precision. Shaders written against
    /// these ops are truly universal — math is the same, precision is silicon.
    ///
    /// For f16/f32/f64: trivial inline wrappers around native operators.
    /// For DF64: routes to `df64_add/df64_mul/etc` from the DF64 core library.
    /// For DF128: routes to `df128_add/df128_mul/etc` (Dekker on f64).
    /// For QF128: routes to `qf128_add/qf128_mul/etc` (Bailey quad-double on f32).
    /// For BF16/FP8: pack/unpack helpers, compute in f32, requantize.
    /// For quantized (Binary/Int2/Q4/Q8): dequantize→f32 compute→quantize
    ///   pattern, not the `op_preamble` abstraction — returns empty preamble.
    ///
    /// All preambles provide identity `op_pack`/`op_unpack` for uniform
    /// array access patterns. DF64 uses these for `vec2<f32>` ↔ `Df64`
    /// conversion; other precisions are identity (compiler eliminates them).
    #[must_use]
    pub fn op_preamble(&self) -> &'static str {
        match self {
            Self::Binary | Self::Int2 | Self::Q4 | Self::Q8 => OP_PREAMBLE_QUANTIZED,
            Self::Fp8E5M2 => OP_PREAMBLE_FP8_E5M2,
            Self::Fp8E4M3 => OP_PREAMBLE_FP8_E4M3,
            Self::Bf16 => OP_PREAMBLE_BF16,
            Self::F16 => OP_PREAMBLE_F16,
            Self::F32 => OP_PREAMBLE_F32,
            Self::F64 => OP_PREAMBLE_F64,
            Self::Df64 => OP_PREAMBLE_DF64,
            Self::Qf128 => OP_PREAMBLE_QF128,
            Self::Df128 => OP_PREAMBLE_DF128,
        }
    }
}

/// f16 operation preamble — native half-precision ops.
///
/// Requires `enable f16;` directive and `SHADER_F16` adapter feature.
/// On hardware without native f16, falls back to f32 with pack/unpack
/// (handled by the compilation pipeline, not this preamble).
const OP_PREAMBLE_F16: &str = r"
// Universal operation preamble — f16 precision (half float, 10-bit mantissa)
enable f16;
alias Scalar = f16;
fn op_add(a: f16, b: f16) -> f16 { return a + b; }
fn op_sub(a: f16, b: f16) -> f16 { return a - b; }
fn op_mul(a: f16, b: f16) -> f16 { return a * b; }
fn op_div(a: f16, b: f16) -> f16 { return a / b; }
fn op_neg(a: f16) -> f16 { return -a; }
fn op_abs(a: f16) -> f16 { return abs(a); }
fn op_max(a: f16, b: f16) -> f16 { return max(a, b); }
fn op_min(a: f16, b: f16) -> f16 { return min(a, b); }
fn op_gt(a: f16, b: f16) -> bool { return a > b; }
fn op_lt(a: f16, b: f16) -> bool { return a < b; }
fn op_ge(a: f16, b: f16) -> bool { return a >= b; }
fn op_le(a: f16, b: f16) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f16 { return f16(v); }
fn op_zero() -> f16 { return f16(0.0); }
fn op_one() -> f16 { return f16(1.0); }
fn op_pack(v: f16) -> f16 { return v; }
fn op_unpack(v: f16) -> f16 { return v; }
";

/// f32 operation preamble — trivial wrappers, compiler inlines everything.
const OP_PREAMBLE_F32: &str = r"
// Universal operation preamble — f32 precision
alias Scalar = f32;
fn op_add(a: f32, b: f32) -> f32 { return a + b; }
fn op_sub(a: f32, b: f32) -> f32 { return a - b; }
fn op_mul(a: f32, b: f32) -> f32 { return a * b; }
fn op_div(a: f32, b: f32) -> f32 { return a / b; }
fn op_neg(a: f32) -> f32 { return -a; }
fn op_abs(a: f32) -> f32 { return abs(a); }
fn op_max(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_min(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_gt(a: f32, b: f32) -> bool { return a > b; }
fn op_lt(a: f32, b: f32) -> bool { return a < b; }
fn op_ge(a: f32, b: f32) -> bool { return a >= b; }
fn op_le(a: f32, b: f32) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f32 { return v; }
fn op_zero() -> f32 { return 0.0; }
fn op_one() -> f32 { return 1.0; }
fn op_pack(v: f32) -> f32 { return v; }
fn op_unpack(v: f32) -> f32 { return v; }
";

/// f64 operation preamble — same structure, f64 types.
const OP_PREAMBLE_F64: &str = r"
// Universal operation preamble — f64 precision
alias Scalar = f64;
fn op_add(a: f64, b: f64) -> f64 { return a + b; }
fn op_sub(a: f64, b: f64) -> f64 { return a - b; }
fn op_mul(a: f64, b: f64) -> f64 { return a * b; }
fn op_div(a: f64, b: f64) -> f64 { return a / b; }
fn op_neg(a: f64) -> f64 { return -a; }
fn op_abs(a: f64) -> f64 { return abs(a); }
fn op_max(a: f64, b: f64) -> f64 { return max(a, b); }
fn op_min(a: f64, b: f64) -> f64 { return min(a, b); }
fn op_gt(a: f64, b: f64) -> bool { return a > b; }
fn op_lt(a: f64, b: f64) -> bool { return a < b; }
fn op_ge(a: f64, b: f64) -> bool { return a >= b; }
fn op_le(a: f64, b: f64) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f64 { return f64(v); }
fn op_zero() -> f64 { return f64(0.0); }
fn op_one() -> f64 { return f64(1.0); }
fn op_pack(v: f64) -> f64 { return v; }
fn op_unpack(v: f64) -> f64 { return v; }
";

/// DF64 operation preamble — routes to `df64_core` library functions.
/// Requires `df64_core.wgsl` + `df64_transcendentals.wgsl` prepended.
const OP_PREAMBLE_DF64: &str = r"
// Universal operation preamble — DF64 precision (f32-pair, ~48-bit mantissa)
alias Scalar = Df64;
alias StorageType = vec2<f32>;
fn op_add(a: Df64, b: Df64) -> Df64 { return df64_add(a, b); }
fn op_sub(a: Df64, b: Df64) -> Df64 { return df64_sub(a, b); }
fn op_mul(a: Df64, b: Df64) -> Df64 { return df64_mul(a, b); }
fn op_div(a: Df64, b: Df64) -> Df64 { return df64_div(a, b); }
fn op_neg(a: Df64) -> Df64 { return df64_neg(a); }
fn op_abs(a: Df64) -> Df64 { return df64_abs(a); }
fn op_max(a: Df64, b: Df64) -> Df64 { if df64_gt(a, b) { return a; } return b; }
fn op_min(a: Df64, b: Df64) -> Df64 { if df64_lt(a, b) { return a; } return b; }
fn op_gt(a: Df64, b: Df64) -> bool { return df64_gt(a, b); }
fn op_lt(a: Df64, b: Df64) -> bool { return df64_lt(a, b); }
fn op_ge(a: Df64, b: Df64) -> bool { return !df64_lt(a, b); }
fn op_le(a: Df64, b: Df64) -> bool { return !df64_gt(a, b); }
fn op_from_f32(v: f32) -> Df64 { return df64_from_f32(v); }
fn op_zero() -> Df64 { return df64_zero(); }
fn op_one() -> Df64 { return df64_from_f32(1.0); }
fn op_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }
fn op_unpack(v: vec2<f32>) -> Df64 { return Df64(v.x, v.y); }
";

/// DF128 operation preamble — routes to `df128_core` library functions.
/// Requires `df128_core.wgsl` prepended. Dekker double-double on f64.
const OP_PREAMBLE_DF128: &str = r"
// Universal operation preamble — DF128 precision (f64-pair, ~104-bit mantissa)
alias Scalar = Df128;
alias StorageType = vec2<f64>;
fn op_add(a: Df128, b: Df128) -> Df128 { return df128_add(a, b); }
fn op_sub(a: Df128, b: Df128) -> Df128 { return df128_sub(a, b); }
fn op_mul(a: Df128, b: Df128) -> Df128 { return df128_mul(a, b); }
fn op_div(a: Df128, b: Df128) -> Df128 { return df128_div(a, b); }
fn op_neg(a: Df128) -> Df128 { return Df128(-a.hi, -a.lo); }
fn op_abs(a: Df128) -> Df128 { return df128_abs(a); }
fn op_max(a: Df128, b: Df128) -> Df128 { if df128_gt(a, b) { return a; } return b; }
fn op_min(a: Df128, b: Df128) -> Df128 { if df128_lt(a, b) { return a; } return b; }
fn op_gt(a: Df128, b: Df128) -> bool { return df128_gt(a, b); }
fn op_lt(a: Df128, b: Df128) -> bool { return df128_lt(a, b); }
fn op_ge(a: Df128, b: Df128) -> bool { return !df128_lt(a, b); }
fn op_le(a: Df128, b: Df128) -> bool { return !df128_gt(a, b); }
fn op_from_f32(v: f32) -> Df128 { return df128_from_f64(f64(v)); }
fn op_zero() -> Df128 { return df128_from_f64(f64(0.0)); }
fn op_one() -> Df128 { return df128_from_f64(f64(1.0)); }
fn op_pack(v: Df128) -> vec2<f64> { return vec2<f64>(v.hi, v.lo); }
fn op_unpack(v: vec2<f64>) -> Df128 { return Df128(v.x, v.y); }
";

/// QF128 operation preamble — routes to `qf128_core` library functions.
/// Requires `qf128_core.wgsl` prepended. Bailey quad-double on f32.
const OP_PREAMBLE_QF128: &str = r"
// Universal operation preamble — QF128 precision (f32-quad, ~96-bit mantissa)
alias Scalar = Qf128;
alias StorageType = vec4<f32>;
fn op_add(a: Qf128, b: Qf128) -> Qf128 { return qf128_add(a, b); }
fn op_sub(a: Qf128, b: Qf128) -> Qf128 { return qf128_sub(a, b); }
fn op_mul(a: Qf128, b: Qf128) -> Qf128 { return qf128_mul(a, b); }
fn op_div(a: Qf128, b: Qf128) -> Qf128 { return qf128_div(a, b); }
fn op_neg(a: Qf128) -> Qf128 { return Qf128(-a.q0, -a.q1, -a.q2, -a.q3); }
fn op_abs(a: Qf128) -> Qf128 { return qf128_abs(a); }
fn op_max(a: Qf128, b: Qf128) -> Qf128 { if qf128_gt(a, b) { return a; } return b; }
fn op_min(a: Qf128, b: Qf128) -> Qf128 { if qf128_lt(a, b) { return a; } return b; }
fn op_gt(a: Qf128, b: Qf128) -> bool { return qf128_gt(a, b); }
fn op_lt(a: Qf128, b: Qf128) -> bool { return qf128_lt(a, b); }
fn op_ge(a: Qf128, b: Qf128) -> bool { return !qf128_lt(a, b); }
fn op_le(a: Qf128, b: Qf128) -> bool { return !qf128_gt(a, b); }
fn op_from_f32(v: f32) -> Qf128 { return qf128_from_f32(v); }
fn op_zero() -> Qf128 { return qf128_from_f32(0.0); }
fn op_one() -> Qf128 { return qf128_from_f32(1.0); }
fn op_pack(v: Qf128) -> vec4<f32> { return vec4<f32>(v.q0, v.q1, v.q2, v.q3); }
fn op_unpack(v: vec4<f32>) -> Qf128 { return Qf128(v.x, v.y, v.z, v.w); }
";

/// BF16 operation preamble — compute in f32, pack/unpack via u32 bit-manip.
const OP_PREAMBLE_BF16: &str = r"
// Universal operation preamble — BF16 precision (bfloat16, 7-bit mantissa)
// Compute is done in f32; pack/unpack convert to/from u32-packed BF16.
alias Scalar = f32;
fn op_add(a: f32, b: f32) -> f32 { return a + b; }
fn op_sub(a: f32, b: f32) -> f32 { return a - b; }
fn op_mul(a: f32, b: f32) -> f32 { return a * b; }
fn op_div(a: f32, b: f32) -> f32 { return a / b; }
fn op_neg(a: f32) -> f32 { return -a; }
fn op_abs(a: f32) -> f32 { return abs(a); }
fn op_max(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_min(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_gt(a: f32, b: f32) -> bool { return a > b; }
fn op_lt(a: f32, b: f32) -> bool { return a < b; }
fn op_ge(a: f32, b: f32) -> bool { return a >= b; }
fn op_le(a: f32, b: f32) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f32 { return v; }
fn op_zero() -> f32 { return 0.0; }
fn op_one() -> f32 { return 1.0; }
fn f32_to_bf16(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let rounding = (bits & 0x0000FFFFu) + 0x00007FFFu + ((bits >> 16u) & 1u);
    return (bits + rounding) >> 16u;
}
fn bf16_to_f32(bits: u32) -> f32 { return bitcast<f32>(bits << 16u); }
fn pack_bf16x2(a: f32, b: f32) -> u32 { return (f32_to_bf16(a) << 16u) | f32_to_bf16(b); }
fn unpack_bf16x2(packed: u32) -> vec2<f32> { return vec2<f32>(bf16_to_f32(packed >> 16u), bf16_to_f32(packed & 0xFFFFu)); }
fn op_pack(v: f32) -> f32 { return v; }
fn op_unpack(v: f32) -> f32 { return v; }
";

/// FP8 E4M3 operation preamble — compute in f32, pack/unpack via u32.
const OP_PREAMBLE_FP8_E4M3: &str = r"
// Universal operation preamble — FP8 E4M3 (3-bit mantissa, 4-bit exponent)
// Compute is done in f32; pack/unpack convert to/from u32-packed FP8.
alias Scalar = f32;
fn op_add(a: f32, b: f32) -> f32 { return a + b; }
fn op_sub(a: f32, b: f32) -> f32 { return a - b; }
fn op_mul(a: f32, b: f32) -> f32 { return a * b; }
fn op_div(a: f32, b: f32) -> f32 { return a / b; }
fn op_neg(a: f32) -> f32 { return -a; }
fn op_abs(a: f32) -> f32 { return abs(a); }
fn op_max(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_min(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_gt(a: f32, b: f32) -> bool { return a > b; }
fn op_lt(a: f32, b: f32) -> bool { return a < b; }
fn op_ge(a: f32, b: f32) -> bool { return a >= b; }
fn op_le(a: f32, b: f32) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f32 { return v; }
fn op_zero() -> f32 { return 0.0; }
fn op_one() -> f32 { return 1.0; }
fn fp8_e4m3_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 7u) & 1u;
    let exp_val = (bits >> 3u) & 0xFu;
    let mant = bits & 0x7u;
    if exp_val == 0u { let val = f32(mant) / 8.0 * exp2(-6.0); if sign != 0u { return -val; } return val; }
    if exp_val == 15u { return bitcast<f32>(0x7FC00000u); }
    let val = (1.0 + f32(mant) / 8.0) * exp2(f32(exp_val) - 7.0);
    if sign != 0u { return -val; } return val;
}
fn f32_to_fp8_e4m3(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let sign = (bits >> 31u) & 1u;
    let abs_v = abs(v);
    if abs_v > 448.0 { return (sign << 7u) | 0x7Eu; }
    if abs_v < exp2(-9.0) { return sign << 7u; }
    let f32_exp = (bits >> 23u) & 0xFFu;
    let f32_mant = bits & 0x7FFFFFu;
    let biased_exp = i32(f32_exp) - 127 + 7;
    let exp_out = u32(clamp(biased_exp, 0, 15));
    let mant_out = (f32_mant >> 20u) & 0x7u;
    return (sign << 7u) | (exp_out << 3u) | mant_out;
}
fn pack_fp8x4(a: f32, b: f32, c: f32, d: f32) -> u32 { return (f32_to_fp8_e4m3(a) << 24u) | (f32_to_fp8_e4m3(b) << 16u) | (f32_to_fp8_e4m3(c) << 8u) | f32_to_fp8_e4m3(d); }
fn unpack_fp8x4(packed: u32) -> vec4<f32> { return vec4<f32>(fp8_e4m3_to_f32((packed >> 24u) & 0xFFu), fp8_e4m3_to_f32((packed >> 16u) & 0xFFu), fp8_e4m3_to_f32((packed >> 8u) & 0xFFu), fp8_e4m3_to_f32(packed & 0xFFu)); }
fn op_pack(v: f32) -> f32 { return v; }
fn op_unpack(v: f32) -> f32 { return v; }
";

/// FP8 E5M2 operation preamble — compute in f32, pack/unpack via u32.
const OP_PREAMBLE_FP8_E5M2: &str = r"
// Universal operation preamble — FP8 E5M2 (2-bit mantissa, 5-bit exponent)
// Compute is done in f32; pack/unpack convert to/from u32-packed FP8.
alias Scalar = f32;
fn op_add(a: f32, b: f32) -> f32 { return a + b; }
fn op_sub(a: f32, b: f32) -> f32 { return a - b; }
fn op_mul(a: f32, b: f32) -> f32 { return a * b; }
fn op_div(a: f32, b: f32) -> f32 { return a / b; }
fn op_neg(a: f32) -> f32 { return -a; }
fn op_abs(a: f32) -> f32 { return abs(a); }
fn op_max(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_min(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_gt(a: f32, b: f32) -> bool { return a > b; }
fn op_lt(a: f32, b: f32) -> bool { return a < b; }
fn op_ge(a: f32, b: f32) -> bool { return a >= b; }
fn op_le(a: f32, b: f32) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f32 { return v; }
fn op_zero() -> f32 { return 0.0; }
fn op_one() -> f32 { return 1.0; }
fn fp8_e5m2_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 7u) & 1u;
    let exp_val = (bits >> 2u) & 0x1Fu;
    let mant = bits & 0x3u;
    if exp_val == 0u { let val = f32(mant) / 4.0 * exp2(-14.0); if sign != 0u { return -val; } return val; }
    if exp_val == 31u { if mant != 0u { return bitcast<f32>(0x7FC00000u); } if sign != 0u { return bitcast<f32>(0xFF800000u); } return bitcast<f32>(0x7F800000u); }
    let val = (1.0 + f32(mant) / 4.0) * exp2(f32(exp_val) - 15.0);
    if sign != 0u { return -val; } return val;
}
fn f32_to_fp8_e5m2(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let sign = (bits >> 31u) & 1u;
    let abs_v = abs(v);
    if abs_v > 57344.0 { return (sign << 7u) | 0x7Bu; }
    if abs_v < exp2(-16.0) { return sign << 7u; }
    let f32_exp = (bits >> 23u) & 0xFFu;
    let f32_mant = bits & 0x7FFFFFu;
    let biased_exp = i32(f32_exp) - 127 + 15;
    let exp_out = u32(clamp(biased_exp, 0, 31));
    let mant_out = (f32_mant >> 21u) & 0x3u;
    return (sign << 7u) | (exp_out << 2u) | mant_out;
}
fn op_pack(v: f32) -> f32 { return v; }
fn op_unpack(v: f32) -> f32 { return v; }
";

/// Quantized format preamble — dequantize→f32 compute→quantize pattern.
///
/// Quantized tiers (Binary, Int2, Q4, Q8) do not use the `op_add`/`op_mul`
/// abstraction. Instead, computation happens via explicit dequantization
/// to f32, native f32 compute, and optional requantization. This preamble
/// provides the f32 compute layer; dequantization helpers are in the
/// format-specific WGSL libraries.
const OP_PREAMBLE_QUANTIZED: &str = r"
// Universal operation preamble — quantized format (compute in f32 after dequantization)
alias Scalar = f32;
fn op_add(a: f32, b: f32) -> f32 { return a + b; }
fn op_sub(a: f32, b: f32) -> f32 { return a - b; }
fn op_mul(a: f32, b: f32) -> f32 { return a * b; }
fn op_div(a: f32, b: f32) -> f32 { return a / b; }
fn op_neg(a: f32) -> f32 { return -a; }
fn op_abs(a: f32) -> f32 { return abs(a); }
fn op_max(a: f32, b: f32) -> f32 { return max(a, b); }
fn op_min(a: f32, b: f32) -> f32 { return min(a, b); }
fn op_gt(a: f32, b: f32) -> bool { return a > b; }
fn op_lt(a: f32, b: f32) -> bool { return a < b; }
fn op_ge(a: f32, b: f32) -> bool { return a >= b; }
fn op_le(a: f32, b: f32) -> bool { return a <= b; }
fn op_from_f32(v: f32) -> f32 { return v; }
fn op_zero() -> f32 { return 0.0; }
fn op_one() -> f32 { return 1.0; }
fn op_pack(v: f32) -> f32 { return v; }
fn op_unpack(v: f32) -> f32 { return v; }
";

/// Inject DF64 pack/unpack helpers for array load/store patterns.
///
/// Converts:
/// - `let x: Df64 = arr[i]` → `let x: Df64 = Df64(arr[i].x, arr[i].y)`
/// - Adds pack helper: `fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }`
///
/// This is injected into the shader source after the DF64 core library.
pub const DF64_PACK_UNPACK: &str = r"
fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }
fn df64_unpack(v: vec2<f32>) -> Df64 { return Df64(v.x, v.y); }
";

/// Shader preparation utilities for f64-canonical WGSL.
///
/// Provides driver-aware patching, polyfill injection, and ILP optimization.
/// Math is written once in f64; these utilities prepare it for hardware dispatch.
///
/// Transitional: driver patching and polyfill injection exist because the
/// sovereign dispatch path (coralReef → coralDriver) is not yet integrated.
/// When coralReef handles compilation end-to-end, these reduce to thin IPC calls.
pub struct ShaderTemplate;

impl ShaderTemplate {
    /// Full `math_f64` polyfill preamble for shaders.
    #[must_use]
    pub fn math_f64_preamble() -> String {
        polyfill::math_f64_preamble()
    }

    /// Prepend `math_f64` preamble to a shader body.
    #[must_use]
    pub fn with_math_f64(shader_body: &str) -> String {
        format!(
            "{}\n\n// User shader:\n{}",
            Self::math_f64_preamble(),
            shader_body
        )
    }

    /// Generate f64 shader with driver-aware exp/log patching (synchronous).
    ///
    /// Uses `needs_f64_exp_log_workaround()` (name-based heuristic). For definitive
    /// detection, async callers should use `device.probe_f64_exp_capable().await` and
    /// pass `!capable` as the workaround flag — probe overrides heuristic when run.
    #[must_use]
    pub fn for_device(shader_body: &str, device: &crate::device::WgpuDevice) -> String {
        Self::for_driver_auto(shader_body, device.needs_f64_exp_log_workaround())
    }

    /// Alias for `for_device`; patches shader for the device.
    #[must_use]
    pub fn for_device_auto(shader_body: &str, device: &crate::device::WgpuDevice) -> String {
        Self::for_driver_auto(shader_body, device.needs_f64_exp_log_workaround())
    }

    /// Patch a WGSL shader's `WARP_SIZE` constant and `@workgroup_size` annotation.
    ///
    /// Replaces `const WARP_SIZE: u32 = 32u;` with the given `wave_size` and
    /// adjusts `@workgroup_size(32, 1, 1)` accordingly. Used to specialise the
    /// single-dispatch Jacobi eigensolve for AMD RDNA2/3 (`wave_size=64`) vs
    /// NVIDIA warp (`wave_size=32`) at shader-compilation time.
    #[must_use]
    pub fn patch_warp_size(shader_body: &str, wave_size: u32) -> String {
        shader_body
            .replace(
                "const WARP_SIZE: u32 = 32u;",
                &format!("const WARP_SIZE: u32 = {wave_size}u;"),
            )
            .replace(
                "@workgroup_size(32, 1, 1)",
                &format!("@workgroup_size({wave_size}, 1, 1)"),
            )
    }

    /// Replace legacy `fossil_f64` calls with native WGSL.
    #[must_use]
    pub fn substitute_fossil_f64(shader_body: &str) -> String {
        polyfill::substitute_fossil_f64(shader_body)
    }

    /// Patch shader for driver (exp/log workaround, f64 polyfills, ILP optimize).
    #[must_use]
    pub fn for_driver_auto(shader_body: &str, needs_exp_log_workaround: bool) -> String {
        // Strip `enable f64;` — naga handles f64 via capability flags, not directives.
        let stripped = shader_body
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");
        // Upgrade any legacy fossil calls to native WGSL builtins first.
        let substituted = polyfill::substitute_fossil_f64(&stripped);
        let patched = polyfill::apply_transcendental_workaround_with_sin_cos(
            &substituted,
            needs_exp_log_workaround,
            false,
        );
        let injected = polyfill::inject_f64_polyfills(&patched, None);
        // ILP-reorders @ilp_region blocks and unrolls @unroll_hint loops.
        // ConservativeModel is used as the latency model (safe fallback when no driver profile).
        crate::shaders::optimizer::WgslOptimizer::default().optimize(&injected)
    }

    /// Variant of `for_driver_auto` that uses device capabilities for
    /// precise ILP scheduling and workaround detection.
    ///
    /// Prefer this when `DeviceCapabilities` is available at shader-compile time.
    #[must_use]
    pub fn for_device_capabilities(
        shader_body: &str,
        needs_exp_log_workaround: bool,
        caps: &crate::device::capabilities::DeviceCapabilities,
    ) -> String {
        let stripped = shader_body
            .lines()
            .filter(|l| l.trim() != "enable f64;")
            .collect::<Vec<_>>()
            .join("\n");
        let use_sin_cos_taylor = caps.needs_sin_f64_workaround() || caps.needs_cos_f64_workaround();
        let substituted = polyfill::substitute_fossil_f64(&stripped);
        let patched = polyfill::apply_transcendental_workaround_with_sin_cos(
            &substituted,
            needs_exp_log_workaround,
            use_sin_cos_taylor,
        );
        let extra_preamble = if use_sin_cos_taylor
            && (patched.contains("sin_f64_safe(") || patched.contains("cos_f64_safe("))
        {
            Some(polyfill::SIN_COS_F64_SAFE_PREAMBLE)
        } else {
            None
        };
        let injected = polyfill::inject_f64_polyfills(&patched, extra_preamble);
        crate::shaders::optimizer::WgslOptimizer::new(caps.latency_model()).optimize(&injected)
    }

    /// Inject only the `math_f64` functions used by the shader.
    #[must_use]
    pub fn with_math_f64_auto(shader_body: &str) -> String {
        use math_f64::F64_FUNCTION_ORDER;
        let mut used_functions: Vec<&str> = F64_FUNCTION_ORDER
            .iter()
            .filter(|func_name| {
                let call_pattern = format!("{func_name}(");
                let call_pattern_space = format!("{func_name} (");
                shader_body.contains(&call_pattern) || shader_body.contains(&call_pattern_space)
            })
            .copied()
            .collect();
        if shader_body.contains("round_f64") && !used_functions.contains(&"round_f64") {
            used_functions.push("round_f64");
        }
        if used_functions.is_empty() {
            return shader_body.to_string();
        }
        format!(
            "{}\n\n// User shader:\n{}",
            polyfill::math_f64_subset(&used_functions),
            shader_body
        )
    }

    /// Generate `math_f64` preamble for a subset of functions.
    #[must_use]
    pub fn math_f64_subset(functions: &[&str]) -> String {
        polyfill::math_f64_subset(functions)
    }

    /// Returns true if shader defines the given function.
    #[must_use]
    pub fn shader_defines_function(shader_body: &str, func_name: &str) -> bool {
        polyfill::shader_defines_function(shader_body, func_name)
    }

    /// Returns true if shader defines the given module-level variable.
    #[must_use]
    pub fn shader_defines_module_var(shader_body: &str, var_name: &str) -> bool {
        polyfill::shader_defines_module_var(shader_body, var_name)
    }

    /// Inject f64 polyfills into shader (no driver-specific patching).
    #[must_use]
    pub fn with_math_f64_safe(shader_body: &str) -> String {
        polyfill::inject_f64_polyfills(shader_body, None)
    }

    /// Alias for `with_math_f64_safe`; injects f64 polyfills.
    #[must_use]
    pub fn with_math_f64_auto_safe(shader_body: &str) -> String {
        polyfill::inject_f64_polyfills(shader_body, None)
    }
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod tests;
