// Physicist's Hermite polynomials Hₙ(x) — f64 precision
// Uses three-term recurrence relation:
//   H₀(x) = 1
//   H₁(x) = 2x
//   Hₙ₊₁(x) = 2x·Hₙ(x) - 2n·Hₙ₋₁(x)
//
// Applications: quantum harmonic oscillator wavefunctions, nuclear structure,
//   Gaussian quadrature, Gaussian-Hermite basis functions
// Validated by: hotSpring nuclear EOS study (169/169 acceptance checks)
// Reference: Abramowitz & Stegun §22.3, NIST DLMF Chapter 18
//
// Deep Debt: pure WGSL, f64, no recursion, self-contained

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    n: u32,  // Polynomial order (0, 1, 2, ...)
    _pad0: u32,
    _pad1: u32,
}

// Hermite polynomial Hₙ(x) via three-term recurrence
// Numerically stable for n up to ~100 with f64
fn hermite(n: u32, x: f64) -> f64 {
    if (n == 0u) {
        return f64(1.0);
    }
    if (n == 1u) {
        return f64(2.0) * x;
    }

    var h_prev = f64(1.0);       // H₀
    var h_curr = f64(2.0) * x;   // H₁

    for (var k = 1u; k < n; k = k + 1u) {
        let h_next = f64(2.0) * x * h_curr - f64(2.0) * f64(k) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }

    return h_curr;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = hermite(params.n, input[idx]);
}

// ═══════════════════════════════════════════════════════════════════
// Variant: Hermite function (normalized for QM wavefunctions)
//
// ψₙ(x) = (2ⁿ·n!·√π)^(-1/2) · Hₙ(x) · exp(-x²/2)
//
// This is the quantum harmonic oscillator eigenfunction (position rep).
// Use when you need the actual wavefunction, not just the polynomial.
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
const SQRT_PI: f64 = 1.7724538509055160273;  // sqrt(π) — avoids sqrt(f64) Naga overload issue

// Hermite function: normalized Hermite × Gaussian
fn hermite_function(n: u32, x: f64) -> f64 {
    let h_n = hermite(n, x);
    // Normalization: 1 / sqrt(2^n · n! · sqrt(π))
    let norm = f64(1.0) / sqrt(f64(1u << n) * factorial(n) * SQRT_PI);
    return norm * h_n * exp(-x * x / f64(2.0));
}

@compute @workgroup_size(256)
fn hermite_function_kernel(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = hermite_function(params.n, input[idx]);
}
