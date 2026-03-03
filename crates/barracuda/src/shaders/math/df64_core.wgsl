// SPDX-License-Identifier: AGPL-3.0-or-later
// Double-float (f32-pair) arithmetic for GPU FP32 core saturation.
//
// Represents f64-precision values as (hi: f32, lo: f32) where val ≈ hi + lo.
// Gives ~48-bit mantissa (~14 decimal digits) using only f32 hardware.
// This runs on the 10,496 FP32 CUDA cores instead of the 164 FP64 units.
//
// Precision hierarchy:
//   f32:  24-bit mantissa,  7 decimal digits
//   df64: 48-bit mantissa, 14 decimal digits  ← this library
//   f64:  53-bit mantissa, 16 decimal digits
//
// FMA optimization (Feb 2026): Uses fma() for error-free products instead
// of Dekker splitting. On Ampere/Ada/RDNA2+, fma() is free-ish (same
// throughput as mul). This eliminates the split() function entirely and
// reduces two_prod from 17 ops to 2 ops. Critical for Krylov solver
// convergence where DF64 mul is the hot path.
//
// Usage: prepend to any shader needing high-throughput approximate f64.
// Use native f64 for precision-critical accumulations and convergence tests.
//
// Absorbed from hotSpring biomeGate FP64 core-streaming discovery (Feb 2026).

struct Df64 {
    hi: f32,
    lo: f32,
}

// ── Constructors ──

fn df64_from_f32(a: f32) -> Df64 {
    return Df64(a, 0.0);
}

fn df64_from_f64(v: f64) -> Df64 {
    let hi = f32(v);
    let lo = f32(v - f64(hi));
    return Df64(hi, lo);
}

fn df64_to_f64(v: Df64) -> f64 {
    return f64(v.hi) + f64(v.lo);
}

fn df64_zero() -> Df64 {
    return Df64(0.0, 0.0);
}

// ── Error-free transformations (Knuth/Dekker) ──

// Two-sum: exact s + e = a + b
fn two_sum(a: f32, b: f32) -> Df64 {
    let s = a + b;
    let v = s - a;
    let e = (a - (s - v)) + (b - v);
    return Df64(s, e);
}

// Two-product: exact p + e = a * b
// Uses FMA for the error term: e = fma(a, b, -p) gives the exact
// rounding error of the product in a single instruction. This replaces
// the 17-op Dekker splitting approach and is both faster and more
// numerically stable on all modern GPUs (Ampere, Ada, RDNA2+, Volta).
fn two_prod(a: f32, b: f32) -> Df64 {
    let p = a * b;
    let e = fma(a, b, -p);
    return Df64(p, e);
}

// ── Core arithmetic ──

fn df64_add(a: Df64, b: Df64) -> Df64 {
    let s = two_sum(a.hi, b.hi);
    let e = a.lo + b.lo;
    let v = two_sum(s.hi, s.lo + e);
    return v;
}

fn df64_sub(a: Df64, b: Df64) -> Df64 {
    return df64_add(a, Df64(-b.hi, -b.lo));
}

fn df64_mul(a: Df64, b: Df64) -> Df64 {
    let p = two_prod(a.hi, b.hi);
    let lo = p.lo + fma(a.hi, b.lo, a.lo * b.hi);
    let r = two_sum(p.hi, lo);
    return r;
}

fn df64_neg(a: Df64) -> Df64 {
    return Df64(-a.hi, -a.lo);
}

fn df64_scale_f32(a: Df64, s: f32) -> Df64 {
    let p = two_prod(a.hi, s);
    let lo = p.lo + a.lo * s;
    return two_sum(p.hi, lo);
}

fn df64_div(a: Df64, b: Df64) -> Df64 {
    let q1 = a.hi / b.hi;
    let r = df64_sub(a, df64_mul(b, df64_from_f32(q1)));
    let q2 = r.hi / b.hi;
    return two_sum(q1, q2);
}
