// SPDX-License-Identifier: AGPL-3.0-or-later
// Generalized Laguerre polynomials L_n^(α)(x) — f64 precision
// Uses three-term recurrence relation:
//   L₀^(α)(x) = 1
//   L₁^(α)(x) = 1 + α - x
//   L_n^(α)(x) = ((2n-1+α-x)·L_{n-1} - (n-1+α)·L_{n-2}) / n
//
// Applications: hydrogen/helium radial wavefunctions, nuclear structure,
//   harmonic oscillator basis (2D/3D), Gamma function computation,
//   exponential fitting, molecular dynamics radial basis
// Validated by: hotSpring nuclear EOS study (169/169 acceptance checks)
// Reference: Abramowitz & Stegun §22.7, NIST DLMF Chapter 18
//
// Deep Debt: pure WGSL, f64, no recursion, self-contained

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    n: u32,       // Polynomial degree
    _pad0: u32,
    _pad1: u32,
    alpha: f64,   // Generalization parameter (0.0 for simple Laguerre)
}

// Generalized Laguerre polynomial L_n^(α)(x) via recurrence
// Numerically stable for n up to ~50 with f64
fn laguerre(n: u32, alpha: f64, x: f64) -> f64 {
    if (n == 0u) {
        return f64(1.0);
    }
    if (n == 1u) {
        return f64(1.0) + alpha - x;
    }

    var l_prev = f64(1.0);              // L₀
    var l_curr = f64(1.0) + alpha - x;  // L₁

    for (var k = 1u; k < n; k = k + 1u) {
        let kf = f64(k);
        // Three-term recurrence: n·Lₙ = (2n-1+α-x)·L_{n-1} - (n-1+α)·L_{n-2}
        let l_next = ((f64(2.0) * kf + f64(1.0) + alpha - x) * l_curr
                      - (kf + alpha) * l_prev) / (kf + f64(1.0));
        l_prev = l_curr;
        l_curr = l_next;
    }

    return l_curr;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = laguerre(params.n, params.alpha, input[idx]);
}

// ═══════════════════════════════════════════════════════════════════
// Variant: Associated Laguerre function (2D harmonic oscillator radial)
//
// R_{n,|m|}(η) = sqrt(n! / (n+|m|)!) · η^(|m|/2) · L_n^|m|(η) · exp(-η/2)
//
// where η = (r/b)² is the dimensionless radial coordinate squared.
// This is the radial eigenfunction for the 2D quantum harmonic oscillator.
// ═══════════════════════════════════════════════════════════════════

// Factorial for small n (max ~25 for f64 accuracy)
fn factorial(n: u32) -> f64 {
    var result = f64(1.0);
    for (var k = 2u; k <= n; k = k + 1u) {
        result = result * f64(k);
    }
    return result;
}

const PI: f64 = 3.14159265358979323846;

struct RadialParams {
    size: u32,
    n: u32,           // Radial quantum number (n_perp)
    abs_m: u32,       // Absolute azimuthal quantum number |m| or |Λ|
    _pad: u32,
    b: f64,           // Oscillator length parameter
}

@group(1) @binding(0) var<uniform> radial_params: RadialParams;
@group(1) @binding(1) var<storage, read> r_input: array<f64>;      // radial coordinate r
@group(1) @binding(2) var<storage, read_write> radial_output: array<f64>;

// 2D harmonic oscillator radial function
@compute @workgroup_size(256)
fn radial_laguerre(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= radial_params.size) {
        return;
    }

    let r = r_input[idx];
    let eta = (r / radial_params.b) * (r / radial_params.b);  // (r/b)²
    let alpha = f64(radial_params.abs_m);

    let n_fact = factorial(radial_params.n);
    let n_plus_m_fact = factorial(radial_params.n + radial_params.abs_m);
    let norm = sqrt(n_fact / (PI * radial_params.b * radial_params.b * n_plus_m_fact));

    let lag = laguerre(radial_params.n, alpha, eta);
    let radial = norm * pow(r / radial_params.b, alpha) * exp(-eta / f64(2.0)) * lag;

    radial_output[idx] = radial;
}
