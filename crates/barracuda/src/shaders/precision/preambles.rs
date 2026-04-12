// SPDX-License-Identifier: AGPL-3.0-or-later
//! WGSL operation preambles for each precision tier.
//!
//! Each preamble provides a universal `op_*` function surface (`op_add`,
//! `op_mul`, `op_pack`, etc.) aliased to the tier's native or emulated type.
//! The generic shader system writes math using these ops; the precision tier
//! selects which preamble to prepend.

/// f16 operation preamble — native half-precision ops.
///
/// Requires `enable f16;` directive and `SHADER_F16` adapter feature.
/// On hardware without native f16, falls back to f32 with pack/unpack
/// (handled by the compilation pipeline, not this preamble).
pub(super) const OP_PREAMBLE_F16: &str = r"
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
pub(super) const OP_PREAMBLE_F32: &str = r"
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
pub(super) const OP_PREAMBLE_F64: &str = r"
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
pub(super) const OP_PREAMBLE_DF64: &str = r"
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
pub(super) const OP_PREAMBLE_DF128: &str = r"
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
pub(super) const OP_PREAMBLE_QF128: &str = r"
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
pub(super) const OP_PREAMBLE_BF16: &str = r"
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
pub(super) const OP_PREAMBLE_FP8_E4M3: &str = r"
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
pub(super) const OP_PREAMBLE_FP8_E5M2: &str = r"
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

/// Quantized format preamble — dequantize->f32 compute->quantize pattern.
///
/// Quantized tiers (Binary, Int2, Q4, Q8) do not use the `op_add`/`op_mul`
/// abstraction. Instead, computation happens via explicit dequantization
/// to f32, native f32 compute, and optional requantization. This preamble
/// provides the f32 compute layer; dequantization helpers are in the
/// format-specific WGSL libraries.
pub(super) const OP_PREAMBLE_QUANTIZED: &str = r"
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
/// - `let x: Df64 = arr[i]` -> `let x: Df64 = Df64(arr[i].x, arr[i].y)`
/// - Adds pack helper: `fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }`
///
/// This is injected into the shader source after the DF64 core library.
pub const DF64_PACK_UNPACK: &str = r"
fn df64_pack(v: Df64) -> vec2<f32> { return vec2<f32>(v.hi, v.lo); }
fn df64_unpack(v: vec2<f32>) -> Df64 { return Df64(v.x, v.y); }
";
