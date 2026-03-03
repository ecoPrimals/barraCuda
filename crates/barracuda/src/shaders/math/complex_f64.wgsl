// complex_f64.wgsl — f64 complex arithmetic library
//
// Prepend this to any WGSL shader that requires complex-number operations.
// All functions operate on vec2<f64> where .x = real part, .y = imaginary part.
//
// Naming convention: c64_*
//   c64_new(re, im)  →  vec2<f64>(re, im)
//
// NVK / Mesa NAK note:
//   c64_exp and c64_phase use sin_f64/cos_f64 (soft polynomial) rather than
//   native sin()/cos() builtins.  Native f64 sin/cos on NVK generate invalid
//   SPIRV ("%N never defined").  ShaderTemplate::inject_missing_math_f64 will
//   auto-inject sin_f64/cos_f64 kernels when this file is compiled via
//   compile_shader_f64 or ShaderTemplate::for_driver_auto.
//
// hotSpring absorption: lattice/complex_f64.rs (v0.5.16, Feb 2026)
// CPU-validated against paper reference implementations.

// ── Constructors ──────────────────────────────────────────────────────────────

fn c64_new(re: f64, im: f64) -> vec2<f64> { return vec2<f64>(re, im); }
fn c64_zero() -> vec2<f64>               { return vec2<f64>(0.0, 0.0); }
fn c64_one()  -> vec2<f64>               { return vec2<f64>(1.0, 0.0); }
fn c64_i()    -> vec2<f64>               { return vec2<f64>(0.0, 1.0); }

// ── Basic arithmetic ──────────────────────────────────────────────────────────

fn c64_add(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x + b.x, a.y + b.y);
}

fn c64_sub(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x - b.x, a.y - b.y);
}

fn c64_mul(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    // (a.x + i·a.y)(b.x + i·b.y) = (a.x·b.x − a.y·b.y) + i(a.x·b.y + a.y·b.x)
    return vec2<f64>(
        a.x * b.x - a.y * b.y,
        a.x * b.y + a.y * b.x,
    );
}

fn c64_conj(a: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x, -a.y);
}

fn c64_scale(a: vec2<f64>, s: f64) -> vec2<f64> {
    return vec2<f64>(a.x * s, a.y * s);
}

// ── Norm and inverse ──────────────────────────────────────────────────────────

fn c64_abs_sq(a: vec2<f64>) -> f64 {
    return a.x * a.x + a.y * a.y;
}

fn c64_abs(a: vec2<f64>) -> f64 {
    return sqrt(c64_abs_sq(a));
}

fn c64_inv(a: vec2<f64>) -> vec2<f64> {
    let denom = c64_abs_sq(a);
    return vec2<f64>(a.x / denom, -a.y / denom);
}

fn c64_div(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return c64_mul(a, c64_inv(b));
}

// ── Exponential ───────────────────────────────────────────────────────────────
// e^(x + iy) = e^x · (cos y + i sin y)
// Uses exp_f64/sin_f64/cos_f64 (soft implementations) so NVK drivers compile
// correctly.  inject_missing_math_f64 auto-injects all three when detected.

fn c64_exp(a: vec2<f64>) -> vec2<f64> {
    let mag = exp_f64(a.x);
    return vec2<f64>(mag * cos_f64(a.y), mag * sin_f64(a.y));
}

// ── Phase factor ──────────────────────────────────────────────────────────────
// e^(i·theta) = cos(theta) + i·sin(theta)
// Uses sin_f64/cos_f64 — native f64 sin/cos broken on NVK (invalid SPIRV).

fn c64_phase(theta: f64) -> vec2<f64> {
    return vec2<f64>(cos_f64(theta), sin_f64(theta));
}
