# barraCuda — Precision Tiers Specification

**Version**: 1.0.0
**Date**: March 11, 2026
**Status**: Specification — defines the full precision tier architecture from Binary to DF128
**Authority**: barraCuda (Layer 1 — math and precision strategy)

---

## 1. Overview

barraCuda owns precision strategy for the ecoPrimals GPU compute stack. Math is
written in f64-canonical WGSL, then compiled to a hardware precision tier that
balances numerical accuracy against throughput. This document defines the
complete precision ladder: 15 discrete tiers spanning 1-bit binary hashing
through ~104-bit mantissa extended precision.

### The Precision-Throughput Continuum

Every tier trades precision for throughput. The relationship is not linear — it
follows the structure of GPU register files:

```
                                     Throughput multiplier (vs F32)
Tier          Mantissa    Storage     ──────────────────────────────
DF128         ~104 bits   2× f64     0.03× (f64 throttled × DD overhead)
QF128         ~96 bits    4× f32     0.12× (4 f32 ops per scalar op)
F64Precise    52 bits     1× f64     0.03× consumer, 0.5× compute
F64           52 bits     1× f64     0.03× consumer, 0.5× compute
DF64          ~48 bits    2× f32     0.4× (DD overhead on f32 cores)
─── F32 ───   23 bits     1× f32     1.0× (baseline)
TF32          10 bits     1× f32     1.0× (NVIDIA tensor cores)
FP16          10 bits     2 per u32   2.0×
BF16          7 bits      2 per u32   2.0×
FP8 E4M3      3 bits      4 per u32   4.0×
FP8 E5M2      2 bits      4 per u32   4.0×
Q8_0          8-bit int   4 per u32   4.0× (+ dequant)
Q4_0          4-bit int   8 per u32   8.0× (+ dequant)
INT2          2-bit       16 per u32  16.0×
Binary        1-bit       32 per u32  32.0×
```

### Architectural Principle

DF64 pairs two FP32 values to **scale up** precision. Sub-FP32 tiers pack
multiple low-precision values into a single u32 register to **scale down**.
Both directions use the same FP32 hardware — the GPU's f32 cores are the
universal substrate.

```
Scale up:    2 f32 → 1 DF64 value     (sacrifice throughput for precision)
             4 f32 → 1 QF128 value    (sacrifice more throughput for more precision)
             2 f64 → 1 DF128 value    (requires f64 hardware)
Baseline:    1 f32 → 1 F32 value
Scale down:  1 u32 → 2 FP16 values    (sacrifice precision for throughput)
             1 u32 → 4 INT8 values    (sacrifice more precision for more throughput)
             1 u32 → 8 INT4 values
             1 u32 → 32 binary values
```

---

## 2. Tier Definitions

### 2.1 Scale-Up Zone

| Tier | Base | Storage | Mantissa | Decimal Digits | Exponent Range | Construction |
|------|------|---------|----------|----------------|----------------|--------------|
| DF128 | f64 | 2× f64 (16 B) | ~104 bits | ~31 | ±1.8×10³⁰⁸ | Dekker double-double on f64 |
| QF128 | f32 | 4× f32 (16 B) | ~96 bits | ~28 | ±3.4×10³⁸ | Bailey quad-double on f32 |
| F64Precise | f64 | 1× f64 (8 B) | 52 bits | ~15.9 | ±1.8×10³⁰⁸ | Native f64, FMA fusion disabled |
| F64 | f64 | 1× f64 (8 B) | 52 bits | ~15.9 | ±1.8×10³⁰⁸ | Native f64 |
| DF64 | f32 | 2× f32 (8 B) | ~48 bits | ~14.4 | ±3.4×10³⁸ | Dekker double-double on f32 |

**DF128 vs QF128**: Both provide ~128-bit-class precision but from different
base types. DF128 (DD on f64) requires native f64 hardware but is simpler and
more precise. QF128 (QD on f32) works on any GPU with FP32 cores — including
consumer hardware with 1:64 FP64:FP32 ratios.

"DF64 + DF64 ≈ DF128?" — There are two valid interpretations:

1. **DD(f64)** = true DF128: apply the Dekker double-double scheme to f64 values.
   Two f64 registers → ~104-bit mantissa. Requires `SHADER_F64`. This is the
   natural extension: the same `df64_core.wgsl` algorithms work with f64
   substituted for f32.

2. **QD(f32)** = QF128: apply Bailey's quad-double scheme to f32 values. Four
   f32 registers → ~96-bit mantissa. No f64 hardware needed. More complex
   renormalization, but universally available.

Both are specified here. DF128 is preferred when f64 hardware is available.
QF128 is the fallback for consumer GPUs.

### 2.2 Baseline

| Tier | Storage | Mantissa | Decimal Digits | Exponent Range |
|------|---------|----------|----------------|----------------|
| F32 | 1× f32 (4 B) | 23 bits | ~7.2 | ±3.4×10³⁸ |

F32 is the universal baseline. Every GPU supports it. The `Precision::F32`
`op_preamble` provides identity wrappers over native operators.

### 2.3 Scale-Down Zone — Floating-Point Formats

| Tier | Storage | Mantissa | Exponent | Range | IEEE Standard |
|------|---------|----------|----------|-------|---------------|
| TF32 | 1× f32 (4 B) | 10 bits | 8 bits | ±3.4×10³⁸ | NVIDIA proprietary |
| FP16 | 2 per u32 | 10 bits | 5 bits | ±65504 | IEEE 754-2008 (binary16) |
| BF16 | 2 per u32 | 7 bits | 8 bits | ±3.4×10³⁸ | Google Brain (de facto) |
| FP8 E4M3 | 4 per u32 | 3 bits | 4 bits | ±448 | OFP E4M3 |
| FP8 E5M2 | 4 per u32 | 2 bits | 5 bits | ±57344 | OFP E5M2 |

**TF32**: NVIDIA tensor core format. Same exponent range as f32 (no overflow
surprise), but only 10-bit mantissa. Cannot be expressed in WGSL directly —
it is an internal tensor core accumulation format. Included here for
completeness; barraCuda treats it as informational.

**FP16**: IEEE 754 half precision. Native WGSL support via `enable f16;` and
`wgpu::Features::SHADER_F16`. Most modern GPUs support this. 2× throughput
over f32 on architectures with dedicated f16 datapaths.

**BF16**: Google Brain float. Same exponent range as f32 (no overflow on
cast-down), 7-bit mantissa. Not natively supported in WGSL — emulated via u32
bit manipulation. Two instructions to pack/unpack. Primary use: ML training
where range matters more than precision.

**FP8 E4M3 / E5M2**: Open Float Point (OFP) 8-bit formats from NVIDIA
Hopper/Ada. E4M3 has more mantissa (inference). E5M2 has more range
(gradients). Emulated via u32 bit ops in WGSL.

### 2.4 Scale-Down Zone — Quantized Integer Formats

| Tier | Packing | Levels | Range | Block Format | Primary Use |
|------|---------|--------|-------|-------------|-------------|
| Q8_0 | 4 per u32 | 256 | -128..127 | 32 values + f16 scale | Quantized inference |
| Q4_0 | 8 per u32 | 16 | -8..7 | 32 values + f16 scale | LLM inference |
| INT2 / Ternary | 16 per u32 | 4 | {-1, 0, +1} or {0..3} | 32 values + f16 scale | Ternary networks |
| Binary | 32 per u32 | 2 | {0, 1} | Raw bitfield | XNOR-net, hashing |

**Block quantization** (Q8_0, Q4_0, INT2): A block of N values (typically 32)
shares a single f16 scale factor. The dequantization formula is:

```
f32_value = scale × (quantized_int - zero_point)
```

For Q4_0 (`zero_point = 8`): `f32_value = scale × (nibble - 8)`
For Q8_0 (`zero_point = 0`, signed): `f32_value = scale × i8_value`

**Binary**: 32 values packed per u32. Dot products become XNOR + popcount —
a single bitwise operation replaces 32 multiplications. Used in binary neural
networks and locality-sensitive hashing.

---

## 3. Scale-Up Mathematics

### 3.1 DF128 — Double-Double on f64

DF128 applies the Dekker (1971) / Knuth (1997) error-free transformation to
f64 values. A DF128 number is a pair `(hi, lo)` of f64 where `value ≈ hi + lo`
and `|lo| ≤ ulp(hi)/2`. This gives ~104-bit mantissa (~2×52 minus overlap).

The algorithms are identical to the existing `df64_core.wgsl`, with `f64`
substituted for `f32`:

```wgsl
// DF128 type: pair of f64
struct Df128 {
    hi: f64,
    lo: f64,
}

// Error-free addition (Knuth two-sum)
fn df128_two_sum(a: f64, b: f64) -> Df128 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df128(s, e);
}

// Error-free multiplication (FMA two-product)
fn df128_two_prod(a: f64, b: f64) -> Df128 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df128(p, e);
}

// DF128 addition
fn df128_add(a: Df128, b: Df128) -> Df128 {
    var s = df128_two_sum(a.hi, b.hi);
    s.lo += a.lo + b.lo;
    s = df128_two_sum(s.hi, s.lo);
    return s;
}

// DF128 subtraction
fn df128_sub(a: Df128, b: Df128) -> Df128 {
    return df128_add(a, Df128(-b.hi, -b.lo));
}

// DF128 multiplication
fn df128_mul(a: Df128, b: Df128) -> Df128 {
    var p = df128_two_prod(a.hi, b.hi);
    p.lo += a.hi * b.lo + a.lo * b.hi;
    p = df128_two_sum(p.hi, p.lo);
    return p;
}

// DF128 division (Newton-Raphson)
fn df128_div(a: Df128, b: Df128) -> Df128 {
    let q1 = a.hi / b.hi;
    let r = df128_sub(a, df128_mul_f64(b, q1));
    let q2 = r.hi / b.hi;
    return df128_two_sum(q1, q2);
}

// Scalar multiply: Df128 × f64
fn df128_mul_f64(a: Df128, s: f64) -> Df128 {
    var p = df128_two_prod(a.hi, s);
    p.lo += a.lo * s;
    p = df128_two_sum(p.hi, p.lo);
    return p;
}

// Conversion
fn df128_from_f64(v: f64) -> Df128 { return Df128(v, 0.0); }
fn df128_to_f64(a: Df128) -> f64 { return a.hi + a.lo; }

// Comparison
fn df128_gt(a: Df128, b: Df128) -> bool {
    return a.hi > b.hi || (a.hi == b.hi && a.lo > b.lo);
}
fn df128_lt(a: Df128, b: Df128) -> bool {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
fn df128_abs(a: Df128) -> Df128 {
    if a.hi < 0.0 { return Df128(-a.hi, -a.lo); }
    return a;
}
```

**Key architectural point**: The `df128_core.wgsl` is a mechanical
substitution of the existing `df64_core.wgsl`. The same Dekker/Knuth
algorithms serve both tiers. The `df128_rewrite` pass in the sovereign
compiler pipeline follows the same pattern as the existing `df64_rewrite`.

**Transcendentals**: `df128_transcendentals.wgsl` follows the same structure
as `df64_transcendentals.wgsl` — Newton-Raphson sqrt, Cody-Waite range
reduction for exp/log, minimax polynomial kernels for sin/cos. The polynomial
coefficients must be extended to ~31-digit precision (source: MPFR or Sollya).

### 3.2 QF128 — Quad-Double on f32

QF128 uses four f32 components `(q0, q1, q2, q3)` ordered by magnitude:
`|q0| > |q1| > |q2| > |q3|` with `|q_{i+1}| < ulp(q_i)/2`. The value is
`q0 + q1 + q2 + q3`. This gives ~96-bit mantissa (~4×24 minus overlap).

Based on Bailey (2003) "A Fortran-90 Double-Double and Quad-Double Package"
and Hida, Li, Bailey (2001) "Algorithms for Quad-Double Precision Floating
Point Arithmetic."

```wgsl
struct Qf128 {
    q0: f32, // most significant
    q1: f32,
    q2: f32,
    q3: f32, // least significant
}

// Renormalize: restore the magnitude-ordering invariant after arithmetic.
// Cascades error terms downward through two-sum operations.
fn qf128_renormalize(a: Qf128) -> Qf128 {
    // Pass 1: cascade upward
    var s3 = a.q3;
    var t = df64_two_sum_f32(a.q2, s3);
    s3 = t.lo; var s2 = t.hi;
    t = df64_two_sum_f32(a.q1, s2);
    s2 = t.lo; var s1 = t.hi;
    t = df64_two_sum_f32(a.q0, s1);
    s1 = t.lo; var s0 = t.hi;

    // Pass 2: cascade downward to flush remaining error
    var r = Qf128(s0, 0.0, 0.0, 0.0);
    t = df64_two_sum_f32(s1, s2);
    r.q1 = t.hi;
    t = df64_two_sum_f32(t.lo, s3);
    r.q2 = t.hi;
    r.q3 = t.lo;
    return r;
}

// Helper: f32 two-sum (same as DF64 core)
fn df64_two_sum_f32(a: f32, b: f32) -> vec2<f32> {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return vec2<f32>(s, e);
}

// QF128 addition: merge two quad-doubles via three-sum cascade
fn qf128_add(a: Qf128, b: Qf128) -> Qf128 {
    // Pairwise two-sum of components, sorted by magnitude
    var s0 = df64_two_sum_f32(a.q0, b.q0);
    var s1 = df64_two_sum_f32(a.q1, b.q1);
    var s2 = df64_two_sum_f32(a.q2, b.q2);
    var s3 = df64_two_sum_f32(a.q3, b.q3);

    // Cascade error terms
    var t1 = df64_two_sum_f32(s0.y, s1.x);
    var t2 = df64_two_sum_f32(s1.y, s2.x);
    var t3 = s2.y + s3.x + s3.y;
    t2 = df64_two_sum_f32(t2.x, t1.y);
    let r2 = t2.x + t3;

    return qf128_renormalize(Qf128(s0.x, t1.x, t2.x, r2));
}

// QF128 multiplication: schoolbook with component pairs
fn qf128_mul(a: Qf128, b: Qf128) -> Qf128 {
    // q0 × q0 (highest-order term)
    var p = df64_two_prod_f32(a.q0, b.q0);
    // Cross terms at decreasing significance
    var c1 = a.q0 * b.q1 + a.q1 * b.q0;
    var c2 = a.q0 * b.q2 + a.q1 * b.q1 + a.q2 * b.q0;
    var c3 = a.q0 * b.q3 + a.q1 * b.q2 + a.q2 * b.q1 + a.q3 * b.q0;

    // Accumulate with error propagation
    var s = df64_two_sum_f32(p.x, c1);
    var t = df64_two_sum_f32(s.y, p.y + c2);
    let r3 = t.y + c3;

    return qf128_renormalize(Qf128(s.x, t.x, t.x, r3));
}

fn df64_two_prod_f32(a: f32, b: f32) -> vec2<f32> {
    let p = a * b;
    let e = fma(a, b, -p);
    return vec2<f32>(p, e);
}

// Conversion
fn qf128_from_f32(v: f32) -> Qf128 { return Qf128(v, 0.0, 0.0, 0.0); }
fn qf128_to_f32(a: Qf128) -> f32 { return a.q0 + a.q1 + a.q2 + a.q3; }
fn qf128_to_f64(a: Qf128) -> f64 {
    return f64(a.q0) + f64(a.q1) + f64(a.q2) + f64(a.q3);
}
```

**QF128 vs DF128 trade-offs**:

| Property | DF128 (DD f64) | QF128 (QD f32) |
|----------|---------------|----------------|
| Mantissa | ~104 bits | ~96 bits |
| Base type | f64 | f32 |
| Requires f64 HW | Yes | No |
| Ops per scalar op | ~10 f64 ops | ~25 f32 ops |
| Consumer GPU throughput | Very slow (1:64 f64) | Slow but feasible |
| Compute GPU throughput | Reasonable (1:2 f64) | Slower than DF128 |
| Implementation complexity | Low (same as DF64) | High (renormalize) |

**Recommendation**: Use DF128 on compute-class GPUs (V100, A100, MI250) with
good f64 support. Use QF128 on consumer GPUs (RTX 3090, RX 7900) where f64 is
throttled. The `PrecisionBrain` routes automatically based on hardware
calibration.

---

## 4. Scale-Down Formats

### 4.1 FP16 — IEEE 754 Half Precision

Native WGSL support via `enable f16;` requires `wgpu::Features::SHADER_F16`.

```wgsl
// Native f16 usage (when SHADER_F16 is available)
@group(0) @binding(0) var<storage, read> input: array<f16>;
@group(0) @binding(1) var<storage, read_write> output: array<f16>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    output[idx] = input[idx] * f16(0.5);
}
```

When `SHADER_F16` is not available, FP16 is emulated via u32 packing:

```wgsl
// Pack two f16 values into one u32
fn pack_f16x2(a: f32, b: f32) -> u32 {
    return pack2x16float(vec2<f32>(a, b));
}

// Unpack u32 into two f32 values (from f16)
fn unpack_f16x2(packed: u32) -> vec2<f32> {
    return unpack2x16float(packed);
}
```

WGSL provides `pack2x16float` / `unpack2x16float` as builtins.

### 4.2 BF16 — Brain Float 16

BF16 is not natively supported in WGSL. It is the upper 16 bits of an f32
(truncated mantissa, same exponent range). Pack/unpack via bit operations:

```wgsl
// f32 → BF16 (truncate lower 16 mantissa bits)
fn f32_to_bf16(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    // Round to nearest even
    let rounding = (bits & 0x0000FFFFu) + 0x00007FFFu + ((bits >> 16u) & 1u);
    return (bits + rounding) >> 16u;
}

// BF16 → f32 (zero-fill lower 16 bits)
fn bf16_to_f32(bits: u32) -> f32 {
    return bitcast<f32>(bits << 16u);
}

// Pack two BF16 values into one u32
fn pack_bf16x2(a: f32, b: f32) -> u32 {
    let hi = f32_to_bf16(a);
    let lo = f32_to_bf16(b);
    return (hi << 16u) | lo;
}

// Unpack u32 into two f32 values (from BF16)
fn unpack_bf16x2(packed: u32) -> vec2<f32> {
    let a = bf16_to_f32(packed >> 16u);
    let b = bf16_to_f32(packed & 0xFFFFu);
    return vec2<f32>(a, b);
}
```

### 4.3 FP8 — 8-bit Floating Point

Two OFP (Open Float Point) variants. Both emulated via u32 packing.

**E4M3** (4-bit exponent, 3-bit mantissa): higher precision, smaller range.
Used for inference weights and activations.

```wgsl
// FP8 E4M3 → f32
fn fp8_e4m3_to_f32(bits: u32) -> f32 {
    let sign = (bits >> 7u) & 1u;
    let exp = (bits >> 3u) & 0xFu;
    let mant = bits & 0x7u;

    if exp == 0u {
        // Subnormal: (-1)^s × 2^(-6) × (0.mant)
        let val = f32(mant) / 8.0 * exp2(-6.0);
        if sign != 0u { return -val; }
        return val;
    }
    if exp == 15u {
        // NaN (E4M3 has no infinity; all-ones exponent with non-zero mantissa = NaN)
        return bitcast<f32>(0x7FC00000u); // quiet NaN
    }
    // Normal: (-1)^s × 2^(exp-7) × (1 + mant/8)
    let val = (1.0 + f32(mant) / 8.0) * exp2(f32(exp) - 7.0);
    if sign != 0u { return -val; }
    return val;
}

// f32 → FP8 E4M3 (with saturation)
fn f32_to_fp8_e4m3(v: f32) -> u32 {
    let bits = bitcast<u32>(v);
    let sign = (bits >> 31u) & 1u;
    let f32_exp = (bits >> 23u) & 0xFFu;
    let f32_mant = bits & 0x7FFFFFu;

    let abs_v = abs(v);
    if abs_v > 448.0 { return (sign << 7u) | 0x7Eu; } // saturate to max
    if abs_v < exp2(-9.0) { return sign << 7u; } // flush to zero

    let biased_exp = i32(f32_exp) - 127 + 7;
    let exp_out = u32(clamp(biased_exp, 0, 15));
    let mant_out = (f32_mant >> 20u) & 0x7u; // truncate to 3 bits
    return (sign << 7u) | (exp_out << 3u) | mant_out;
}

// Pack 4 FP8 values into one u32
fn pack_fp8x4(a: f32, b: f32, c: f32, d: f32) -> u32 {
    return (f32_to_fp8_e4m3(a) << 24u)
         | (f32_to_fp8_e4m3(b) << 16u)
         | (f32_to_fp8_e4m3(c) << 8u)
         | f32_to_fp8_e4m3(d);
}

// Unpack u32 into 4 f32 values
fn unpack_fp8x4(packed: u32) -> vec4<f32> {
    return vec4<f32>(
        fp8_e4m3_to_f32((packed >> 24u) & 0xFFu),
        fp8_e4m3_to_f32((packed >> 16u) & 0xFFu),
        fp8_e4m3_to_f32((packed >> 8u) & 0xFFu),
        fp8_e4m3_to_f32(packed & 0xFFu),
    );
}
```

**E5M2**: Same structure but with 5-bit exponent and 2-bit mantissa. Wider
range (±57344) but coarser. Used for gradient communication in distributed
training. Implementation follows the same pattern as E4M3 with adjusted
bias and mantissa width.

### 4.4 INT4 — 4-bit Integer (Block Quantized)

Packs 8 signed INT4 values per u32. Used with block quantization (Q4_0
format — 32 values share one f16 scale).

```wgsl
// Extract one INT4 value from packed u32
fn extract_int4(packed: u32, index: u32) -> f32 {
    let shift = index * 4u;
    let nibble = (packed >> shift) & 0xFu;
    // Sign-extend: if bit 3 set, value is negative (-8..+7)
    let signed = i32(nibble) - i32((nibble & 8u) << 1u);
    return f32(signed);
}

// Dequantize a block of 8 INT4 values from one u32
fn dequant_int4_block(packed: u32, scale: f32) -> array<f32, 8> {
    var out: array<f32, 8>;
    for (var i = 0u; i < 8u; i++) {
        out[i] = scale * extract_int4(packed, i);
    }
    return out;
}

// INT4 block dot product: 8 multiplies from one u32 load
fn int4_dot8(a_packed: u32, b_packed: u32, scale_a: f32, scale_b: f32) -> f32 {
    var acc: f32 = 0.0;
    for (var i = 0u; i < 8u; i++) {
        acc += extract_int4(a_packed, i) * extract_int4(b_packed, i);
    }
    return acc * scale_a * scale_b;
}
```

### 4.5 INT2 / Ternary — 2-bit Values

Packs 16 ternary values {-1, 0, +1} per u32 using 2 bits each.

```wgsl
// Extract one INT2 value from packed u32 (ternary: 0=-1, 1=0, 2=+1, 3=unused)
fn extract_ternary(packed: u32, index: u32) -> f32 {
    let shift = index * 2u;
    let bits = (packed >> shift) & 3u;
    return f32(i32(bits) - 1); // 0→-1, 1→0, 2→+1
}

// Ternary dot product: 16 ternary multiplies from one u32 pair
fn ternary_dot16(a_packed: u32, b_packed: u32, scale: f32) -> f32 {
    var acc: f32 = 0.0;
    for (var i = 0u; i < 16u; i++) {
        acc += extract_ternary(a_packed, i) * extract_ternary(b_packed, i);
    }
    return acc * scale;
}
```

### 4.6 Binary — 1-bit Values

32 binary values per u32. Dot products via XNOR + popcount.

```wgsl
// Binary dot product: XNOR + popcount replaces 32 multiplications
fn binary_dot32(a: u32, b: u32) -> f32 {
    let xnor = ~(a ^ b); // XNOR: 1 where bits match
    let matches = countOneBits(xnor);
    // matches counts agreement; 32 - matches counts disagreement
    // dot = matches - (32 - matches) = 2*matches - 32
    return f32(i32(matches) * 2 - 32);
}
```

---

## 5. Quantization Block Formats

### 5.1 Existing Formats (already implemented)

| Format | Block Size | Storage | Compression | Scale Type |
|--------|-----------|---------|-------------|------------|
| Q4_0 | 32 values | 18 B/block | 7.1× | f16 absmax |
| Q8_0 | 32 values | 34 B/block | 3.8× | f16 absmax |

These are implemented in `shaders/quantized.rs` with CPU reference
implementations and WGSL GEMV shaders.

### 5.2 Future Formats (specification only)

| Format | Block Size | Storage | Compression | Scale Type | Notes |
|--------|-----------|---------|-------------|------------|-------|
| Q4_1 | 32 values | 20 B/block | 6.4× | f16 scale + f16 min | Asymmetric |
| Q5_0 | 32 values | 22 B/block | 5.8× | f16 absmax | 5-bit, better quality than Q4 |
| Q5_1 | 32 values | 24 B/block | 5.3× | f16 scale + f16 min | 5-bit asymmetric |
| Q2_K | 256 values | 84 B/block | 12.2× | Super-block with sub-scales | 2-bit K-quant |
| Q3_K | 256 values | 110 B/block | 9.3× | Super-block with sub-scales | 3-bit K-quant |
| Q4_K | 256 values | 144 B/block | 7.1× | Super-block with sub-scales | 4-bit K-quant |
| Q6_K | 256 values | 210 B/block | 4.9× | Super-block with sub-scales | 6-bit K-quant |

K-quant formats (from GGML) use hierarchical scale quantization:
a super-block of 256 values is divided into sub-blocks, each with its own
scale factor. The sub-block scales are themselves quantized. This improves
accuracy at the same bit-width by allowing finer-grained scale adaptation.

---

## 6. Reference Implementation Sources

### 6.1 Verification Strategy Per Tier

| Tier | Gold Standard | Rust Crate | Verification Method |
|------|--------------|------------|---------------------|
| DF128 | MPFR 256-bit | `rug` (MPFR bindings) | Compute at 256-bit, round to 104-bit, compare ULP error |
| QF128 | MPFR 256-bit | `rug` | Compute at 256-bit, round to 96-bit, compare ULP error |
| F64 | IEEE 754, `libm` | `std::f64` | Standard conformance, ULP tables from Cody & Waite |
| DF64 | MPFR 128-bit | `rug` | Already tested in `precision_tests.rs` |
| F32 | IEEE 754 | `std::f32` | Standard conformance |
| FP16 | IEEE 754-2008 | `half` | Bit-exact round-trip, special value handling |
| BF16 | Google Brain spec | `half` | Bit-exact pack/unpack, round-to-nearest-even |
| FP8 | OFP specification | Manual bit tables | Exhaustive (only 256 values per variant) |
| Q4_0/Q8_0 | GGML `ggml-quants.c` | In-tree CPU reference | Already tested |
| INT2/Binary | Exact integer math | Trivial | Exhaustive (4 or 2 values) |

**Note**: `rug` requires GMP/MPFR C libraries. For pure-Rust CI, use `dashu`
(pure Rust arbitrary precision) or pre-computed reference tables embedded as
constants. Reference tables are preferred for barraCuda's ecoBin compliance.

### 6.2 Canonical Test Constants

High-precision reference values for cross-tier validation. All values below
are exact to 35+ decimal digits (sourced from MPFR/Mathematica):

```
π   = 3.14159265358979323846264338327950288...
e   = 2.71828182845904523536028747135266250...
ln2 = 0.69314718055994530941723212145817657...
√2  = 1.41421356237309504880168872420969808...
γ   = 0.57721566490153286060651209008240243... (Euler-Mascheroni)
φ   = 1.61803398874989484820458683436563812... (golden ratio)
```

Expected precision per tier:

| Tier | π correct digits | ULP bound for exp(1.0) |
|------|-----------------|----------------------|
| DF128 | 31 | 1 ULP of f64-pair |
| QF128 | 28 | 1 ULP of f32-quad |
| F64 | 15 | 1 ULP |
| DF64 | 14 | 2-4 ULP (Dekker error) |
| F32 | 7 | 1 ULP |
| FP16 | 3 | 1 ULP |
| BF16 | 2 | 1 ULP |
| FP8 E4M3 | 1 | 1 ULP (only 8 mantissa values) |

### 6.3 GPU Verification Pipeline

For each tier, the verification pipeline is:

1. **CPU reference**: Compute expected result at higher precision (MPFR or next
   tier up), truncate to tier precision
2. **GPU compute**: Run the tier's shader on test input
3. **Readback and compare**: Download result, compare against CPU reference
   within tier-appropriate tolerance
4. **ULP analysis**: Report max ULP error, mean ULP error, and worst-case input

---

## 7. Tolerance Mapping

Each precision tier maps to the existing tolerance registry
(`tolerances.rs`) with tier-specific absolute and relative tolerances:

| Tier | abs_tol | rel_tol | Tolerance Category | Notes |
|------|---------|---------|-------------------|-------|
| DF128 | 1e-28 | 1e-26 | MACHINE | Near-arbitrary precision |
| QF128 | 1e-24 | 1e-22 | MACHINE | 4× f32 Dekker error |
| F64Precise | 1e-14 | 1e-12 | MACHINE | Gold standard |
| F64 | 1e-12 | 1e-10 | MACHINE | FMA may introduce 1 ULP |
| DF64 | 1e-10 | 1e-8 | ACCUMULATION | Dekker error accumulation |
| F32 | 1e-5 | 1e-4 | ACCUMULATION | Standard |
| FP16 | 1e-2 | 5e-2 | TRANSCENDENTAL | 10-bit mantissa |
| BF16 | 5e-2 | 1e-1 | STATISTICAL | 7-bit mantissa |
| FP8 E4M3 | 0.5 | 0.25 | STOCHASTIC | 3-bit mantissa |
| FP8 E5M2 | 1.0 | 0.5 | STOCHASTIC | 2-bit mantissa |
| Q8_0 | varies | varies | QUANTIZED | Depends on scale factor |
| Q4_0 | varies | varies | QUANTIZED | Depends on scale factor |
| INT2 | exact | exact | DETERMINISM | 3 discrete values |
| Binary | exact | exact | DETERMINISM | 2 discrete values |

The `PrecisionBrain` uses these tolerances to validate that a given tier is
producing correct results for a domain. If results exceed the tier's tolerance
bounds, the brain escalates to the next higher tier.

---

## 8. Hardware Requirements

### 8.1 wgpu Feature Mapping

| Tier | wgpu Feature | Universally Available | Notes |
|------|-------------|----------------------|-------|
| DF128 | `SHADER_F64` | No (compute GPUs) | Requires native f64 |
| QF128 | None | Yes | Pure f32 arithmetic |
| F64Precise | `SHADER_F64` | No | Native f64 + FMA control |
| F64 | `SHADER_F64` | No | Native f64 |
| DF64 | None | Yes | Pure f32 arithmetic |
| F32 | None | Yes | Universal |
| FP16 | `SHADER_F16` | Mostly (modern GPUs) | Native half precision |
| BF16 | None | Yes | Emulated via u32 bit ops |
| FP8 | None | Yes | Emulated via u32 bit ops |
| Q8_0 / Q4_0 | None | Yes | Integer ops on u32 |
| INT2 / Binary | None | Yes | Bitwise ops on u32 |

### 8.2 Precision Tier Availability Matrix

```
                        Consumer GPU    Compute GPU     CPU (fallback)
                        (RTX 3090)      (A100/MI250)    (Rust f64)
DF128                   Very slow*      Moderate        Fast
QF128                   Slow            Slow            Fast
F64Precise              Very slow*      Moderate        Fast
F64                     Very slow*      Moderate        Fast
DF64                    Fast            Fast            Fast
F32                     Baseline        Baseline        Baseline
FP16                    2× baseline     2× baseline     Emulated
BF16                    ~baseline       ~baseline       Emulated
FP8                     ~baseline       ~baseline       Emulated
Q8_0                    ~2× baseline    ~2× baseline    Emulated
Q4_0                    ~4× baseline    ~4× baseline    Emulated
INT2                    ~8× baseline    ~8× baseline    Emulated
Binary                  ~16× baseline   ~16× baseline   Emulated

* Consumer GPUs throttle f64 at 1:32 to 1:64 ratio vs f32
```

---

## 9. Integration with Existing Infrastructure

### 9.1 Precision Enum Extension

The `Precision` enum in `shaders/precision/mod.rs` extends to:

```
Precision::Binary    → 1 bit per value
Precision::Int2      → 2 bits per value
Precision::Q4        → 4-bit block quantized
Precision::Q8        → 8-bit block quantized
Precision::Fp8E4M3   → 8-bit float
Precision::Fp8E5M2   → 8-bit float
Precision::Bf16      → 16-bit bfloat
Precision::F16       → 16-bit IEEE half
Precision::F32       → 32-bit float (existing)
Precision::Df64      → 48-bit emulated (existing)
Precision::F64       → 64-bit float (existing)
Precision::Qf128     → 96-bit quad-f32
Precision::Df128     → 104-bit double-f64
```

### 9.2 PrecisionTier Enum Extension

The `PrecisionTier` enum in `device/precision_tier.rs` extends to:

```
PrecisionTier::Quantized4  → Q4_0 block quantized
PrecisionTier::Quantized8  → Q8_0 block quantized
PrecisionTier::F16         → IEEE half precision
PrecisionTier::F32         → 32-bit (existing)
PrecisionTier::DF64        → double-float f32 (existing)
PrecisionTier::F64         → native f64 (existing)
PrecisionTier::F64Precise  → f64 FMA-free (existing)
PrecisionTier::QF128       → quad-float f32
PrecisionTier::DF128       → double-float f64
```

### 9.3 DType Enum Extension

The `DType` enum in `unified_math.rs` extends to:

```
DType::Binary → packed 1-bit
DType::I2     → packed 2-bit
DType::I4     → packed 4-bit
DType::I8     → signed 8-bit
DType::F8E4M3 → 8-bit float
DType::F8E5M2 → 8-bit float
DType::Bf16   → bfloat16
DType::F16    → IEEE half
DType::F32    → 32-bit float (existing)
DType::F64    → 64-bit float (existing)
DType::I32    → 32-bit int (existing)
DType::I64    → 64-bit int (existing)
DType::U32    → 32-bit uint (existing)
DType::U64    → 64-bit uint (existing)
DType::Bool   → boolean (existing)
```

### 9.4 op_preamble Pattern

The existing `op_preamble` pattern (`op_add`, `op_mul`, etc.) extends
naturally to new tiers. Each tier provides its own preamble that maps abstract
operations to concrete implementations:

- **DF128**: Routes to `df128_add`, `df128_mul`, etc.
- **QF128**: Routes to `qf128_add`, `qf128_mul`, etc.
- **FP16**: Routes to native `f16` operators (when available) or emulated
- **Quantized tiers**: Not applicable — quantized compute follows the
  dequantize→f32 compute→quantize pattern, not the `op_preamble` abstraction

### 9.5 PrecisionBrain Extension

The `PrecisionBrain` domain routing extends to include scale-down tiers for
inference workloads. New `PhysicsDomain` variants:

```
PhysicsDomain::Inference    → Q4_0 → Q8_0 → F16 → F32
PhysicsDomain::Training     → BF16 → F32 → DF64
PhysicsDomain::Hashing      → Binary
```

### 9.6 Shader Pipeline

Scale-up tiers follow the existing pipeline:

```
WGSL (f64-canonical) → naga → df128_rewrite → sovereign compiler → wgpu
                               ^^^^^^^^^^^^^^^^
                               same pattern as df64_rewrite
```

Scale-down tiers use pack/unpack shaders without a rewrite pass — the
quantization is explicit in the algorithm, not transparent like DF64/DF128.

---

## 10. Implementation Roadmap

### Phase 1 — FP16 (lowest lift, highest demand)

- Enable `SHADER_F16` feature detection in `WgpuDevice`
- Add `Precision::F16` with native `f16` op_preamble
- Add emulated FP16 fallback via `pack2x16float` / `unpack2x16float`
- Tolerance tier: `TRANSCENDENTAL` (1e-2 abs)
- Tests: round-trip accuracy, GEMV throughput, mixed-precision accumulation
- Reference: `half` crate for CPU verification

### Phase 2 — BF16 (ML training support)

- Add `Precision::Bf16` with u32 bit-manipulation pack/unpack
- WGSL helper functions: `f32_to_bf16`, `bf16_to_f32`, `pack_bf16x2`, `unpack_bf16x2`
- Tolerance tier: `STATISTICAL` (5e-2 abs)
- Tests: round-trip accuracy, range preservation (f32 exponent range maintained)
- Reference: `half::bf16` for CPU verification

### Phase 3 — DF128 (peak precision, scientific computing)

- New `df128_core.wgsl` shader (mechanical port of `df64_core.wgsl`)
- New `df128_transcendentals.wgsl` (extended polynomial coefficients from MPFR)
- `df128_rewrite` pass (extend existing `df64_rewrite`)
- `Precision::Df128` with op_preamble
- Tolerance tier: `MACHINE` (1e-28 abs)
- Tests: pi digits, transcendental accuracy, cancellation resistance
- Reference: pre-computed MPFR tables embedded as test constants

### Phase 4 — QF128 (universal high precision)

- New `qf128_core.wgsl` shader (Bailey quad-double)
- Renormalization and cascade algorithms
- `Precision::Qf128` with op_preamble
- Tolerance tier: `MACHINE` (1e-24 abs)
- Tests: accuracy vs DF128, consumer GPU execution, renormalization stability
- Reference: pre-computed MPFR tables, cross-validation against DF128

### Phase 5 — FP8 variants (ML inference)

- E4M3 and E5M2 pack/unpack WGSL shaders
- Quantization and dequantization with saturation
- GEMV with on-the-fly FP8 dequantization
- Tests: exhaustive (256 values), round-trip, GEMV accuracy
- Reference: OFP specification, bit-exact tables

### Phase 6 — INT2/Binary (extreme quantization)

- Ternary pack/unpack WGSL shaders
- Binary XNOR+popcount dot product
- Tests: exhaustive, GEMV, hashing correctness
- Reference: exact integer math

### Phase 7 — K-quant formats (advanced quantization)

- Q2_K, Q3_K, Q4_K, Q6_K super-block formats
- Hierarchical scale quantization
- Tests: accuracy vs GGML reference, GEMV throughput
- Reference: GGML `ggml-quants.c`

---

## 11. Provenance

### Academic References

- Dekker, T.J. (1971). "A floating-point technique for extending the
  available precision." *Numerische Mathematik*, 18(3), 224-242.

- Knuth, D.E. (1997). *The Art of Computer Programming, Volume 2:
  Seminumerical Algorithms*, 3rd edition. Section 4.2.2, Theorem B.

- Bailey, D.H. (2003). "A Fortran-90 double-double and quad-double
  package." Lawrence Berkeley National Laboratory Technical Report.

- Hida, Y., Li, X.S., Bailey, D.H. (2001). "Algorithms for quad-double
  precision floating point arithmetic." *15th IEEE Symposium on Computer
  Arithmetic*, pp. 155-162.

- Cody, W.J. and Waite, W. (1980). *Software Manual for the Elementary
  Functions*. Prentice-Hall.

- IEEE 754-2008. "IEEE Standard for Floating-Point Arithmetic."

- Micikevicius, P. et al. (2022). "FP8 Formats for Deep Learning."
  arXiv:2209.05433. (OFP E4M3/E5M2 specification.)

- Dettmers, T. et al. (2022). "LLM.int8(): 8-bit Matrix Multiplication
  for Transformers at Scale." arXiv:2208.07339.

- Courbariaux, M. et al. (2016). "Binarized Neural Networks."
  *Advances in Neural Information Processing Systems*, 29.

### Software References

- GGML / llama.cpp: Q4_0, Q8_0, K-quant block formats
  (https://github.com/ggerganov/llama.cpp)

- `half` crate: FP16 and BF16 Rust reference (https://crates.io/crates/half)

- MPFR: GNU Multiple Precision Floating-Point Reliable Library
  (https://www.mpfr.org/)

- `rug` crate: Rust bindings for GMP/MPFR
  (https://crates.io/crates/rug)

- `dashu` crate: Pure Rust arbitrary precision
  (https://crates.io/crates/dashu)

---

## 12. Summary

The precision ladder from Binary to DF128 spans 7 orders of magnitude in
mantissa bits (1 to 104) and 5 orders of magnitude in throughput (0.03× to
32× of F32 baseline). Every tier is implementable in WGSL on current wgpu
hardware, with the sole exception of TF32 (NVIDIA tensor core internal
format).

The key architectural insight: DF64 (existing) and DF128 (new) share identical
Dekker/Knuth algorithms — only the base type changes. QF128 extends this to
GPUs without f64 hardware. Scale-down tiers use integer packing and
dequantization, which are standard GPU compute patterns.

barraCuda's existing infrastructure — `Precision` enum, `PrecisionTier`,
`op_preamble`, `PrecisionBrain`, `df64_rewrite`, and the tolerance registry —
provides natural extension points for every tier defined here.
