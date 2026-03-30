// SPDX-License-Identifier: AGPL-3.0-or-later
//
// stable_f64.wgsl — Numerically stable GPU special function primitives (f64)
//
// Cross-spring P1 (ISSUE-011): functions that avoid catastrophic cancellation
// in the nine known cancellation families documented in
// wateringHole/GPU_F64_NUMERICAL_STABILITY.md.
//
// These are pure f64 polyfills for use in GPU shaders where the naive
// implementation would lose significant precision.

/// log(1 + x) without cancellation for small x.
///
/// For |x| < 1e-4, uses Taylor series: x - x²/2 + x³/3 - x⁴/4.
/// For larger x, uses the Kahan compensated form: x * log(1+x) / ((1+x) - 1).
fn log1p_f64(x: f64) -> f64 {
    let ax = abs(x);
    if ax < 1e-15 {
        return x;
    }
    if ax < 1e-4 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        return x - x2 * 0.5 + x3 / 3.0 - x4 * 0.25;
    }
    let u = 1.0 + x;
    if u == 1.0 {
        return x;
    }
    return log(u) * x / (u - 1.0);
}

/// exp(x) - 1 without cancellation for small x.
///
/// For |x| < 1e-5, uses Taylor series: x + x²/2 + x³/6 + x⁴/24.
/// For larger x, uses the compensated form: (exp(x) - 1) via direct subtraction
/// only when exp(x) is far from 1.
fn expm1_f64(x: f64) -> f64 {
    let ax = abs(x);
    if ax < 1e-15 {
        return x;
    }
    if ax < 1e-5 {
        let x2 = x * x;
        let x3 = x2 * x;
        let x4 = x3 * x;
        return x + x2 * 0.5 + x3 / 6.0 + x4 / 24.0;
    }
    let e = exp(x);
    if e == 1.0 {
        return x;
    }
    return (e - 1.0) * x / log(e);
}

/// Complementary error function erfc(x) = 1 - erf(x).
///
/// Uses Abramowitz & Stegun 7.1.26 rational approximation for x >= 0.
/// For x < 0: erfc(x) = 2 - erfc(-x).
/// Avoids catastrophic cancellation that occurs when computing 1 - erf(x)
/// for large x where erf(x) ≈ 1.
///
/// WGSL disallows recursion; the non-negative case is in a helper.
fn erfc_x_nonneg_f64(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027
        + t * 1.061405429))));

    return poly * exp(-x * x);
}

fn erfc_f64(x: f64) -> f64 {
    if x < 0.0 {
        return 2.0 - erfc_x_nonneg_f64(-x);
    }
    return erfc_x_nonneg_f64(x);
}

/// Stable Bessel J₀(x) - 1 for small x.
///
/// Direct computation of J₀(x) - 1 via power series avoids cancellation
/// when J₀(x) ≈ 1 for small x.
fn bessel_j0_minus1_f64(x: f64) -> f64 {
    let x2 = x * x;
    if abs(x) < 0.5 {
        let x4 = x2 * x2;
        let x6 = x4 * x2;
        return -x2 / 4.0 + x4 / 64.0 - x6 / 2304.0;
    }
    // For larger x, let the caller decide (this function is for the small-x regime).
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x6 * x2;
    return -x2 / 4.0 + x4 / 64.0 - x6 / 2304.0 + x8 / 147456.0;
}

// Entry point for validation — computes all four functions and writes results.
struct StableParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: StableParams;

@compute @workgroup_size(256)
fn log1p_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    output[idx] = log1p_f64(input[idx]);
}

@compute @workgroup_size(256)
fn expm1_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    output[idx] = expm1_f64(input[idx]);
}

@compute @workgroup_size(256)
fn erfc_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    output[idx] = erfc_f64(input[idx]);
}

@compute @workgroup_size(256)
fn bessel_j0m1_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    output[idx] = bessel_j0_minus1_f64(input[idx]);
}
