// SPDX-License-Identifier: AGPL-3.0-or-later
// ============================================================================
// math_f64_fossils.wgsl — superseded f64 software implementations
// ============================================================================
//
// FOSSIL FUNCTIONS: probe-confirmed native WGSL f64 builtins on ALL
// SHADER_F64 hardware via Vulkan (Feb 18 2026, bench_f64_builtins —
// RTX 3090 + RX 6950 XT):
//
//   abs(f64)    min(f64,f64)  max(f64,f64)  clamp(f64,f64,f64)
//   sqrt(f64)   floor(f64)    ceil(f64)     round(f64)
//   sign(f64)   fract(f64)
//
// These software implementations are retained as reference / emergency
// fallback for future edge-case GPU profiling. New shader code MUST use
// native WGSL builtins directly. ShaderTemplate will NOT inject these.
// See F64_FOSSIL_FUNCTIONS in math_f64.rs.
//
// To rewrite legacy fossil calls to native equivalents:
//   ShaderTemplate::substitute_fossil_f64()
// ============================================================================

// Requires f64_const from math_f64.wgsl

// 🦴 FOSSIL — use native abs(x)
fn abs_f64(x: f64) -> f64 {
    if (x < f64_const(x, 0.0)) {
        return -x;
    }
    return x;
}

// 🦴 FOSSIL — use native sign(x)
fn sign_f64(x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    if (x > zero) {
        return f64_const(x, 1.0);
    }
    if (x < zero) {
        return f64_const(x, -1.0);
    }
    return zero;
}

// 🦴 FOSSIL — use native floor(x)
fn floor_f64(x: f64) -> f64 {
    let i = i32(x);
    let fi = f64(i);
    if (x < fi) {
        return fi - f64_const(x, 1.0);
    }
    return fi;
}

// 🦴 FOSSIL — use native ceil(x)
fn ceil_f64(x: f64) -> f64 {
    let i = i32(x);
    let fi = f64(i);
    if (x > fi) {
        return fi + f64_const(x, 1.0);
    }
    return fi;
}

// 🦴 FOSSIL — use native round(x)
fn round_f64(x: f64) -> f64 {
    return floor_f64(x + f64_const(x, 0.5));
}

// 🦴 FOSSIL — use native fract(x)
fn fract_f64(x: f64) -> f64 {
    return x - floor_f64(x);
}

// 🦴 FOSSIL — use native min(a, b)
fn min_f64(a: f64, b: f64) -> f64 {
    if (a < b) { return a; }
    return b;
}

// 🦴 FOSSIL — use native max(a, b)
fn max_f64(a: f64, b: f64) -> f64 {
    if (a > b) { return a; }
    return b;
}

// 🦴 FOSSIL — use native clamp(x, lo, hi)
fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    return min_f64(max_f64(x, lo), hi);
}

// ============================================================================
// SQUARE ROOT — Newton-Raphson (kept as fossil; native sqrt(f64) preferred)
// ============================================================================

// 🦴 FOSSIL — use native sqrt(x) on all SHADER_F64 hardware.
// Probe-confirmed native on RTX 3090 (PTXAS) and RX 6950 XT (ACO) Feb 2026.
// Newton-Raphson implementation retained for edge-case GPU profiling only.
fn sqrt_f64(x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    if (x <= zero) {
        return zero;
    }
    
    // Initial estimate using f32 sqrt (via bit manipulation approximation)
    // For robustness, use a simple initial guess based on magnitude
    var y = x;
    
    // Scale to reasonable range for initial guess
    var scale = f64_const(x, 1.0);
    let large = f64_const(x, 1e32);
    let small = f64_const(x, 1e-32);
    
    if (x > large) {
        y = x / large;
        scale = f64_const(x, 1e16);
    } else if (x < small) {
        y = x * large;
        scale = f64_const(x, 1e-16);
    }
    
    // Initial guess: y^0.5 ≈ y / 2 for y near 1 (crude but converges)
    var r = (y + f64_const(x, 1.0)) / f64_const(x, 2.0);
    
    // Newton-Raphson iterations
    let half = f64_const(x, 0.5);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    
    return r * scale;
}
