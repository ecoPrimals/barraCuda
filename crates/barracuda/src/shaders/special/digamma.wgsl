// SPDX-License-Identifier: AGPL-3.0-or-later
// Digamma function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
// Uses reflection + recurrence + asymptotic expansion
//
// For x < 0: ψ(x) = ψ(1-x) - π·cot(πx)  [reflection formula]
// For x > 0, x < 6: use recurrence ψ(x+1) = ψ(x) + 1/x
// For x ≥ 6: asymptotic expansion ψ(x) ≈ ln(x) - 1/(2x) - Σ B₂ₖ/(2k·x²ᵏ)
//
// Applications: Fisher information, Bayesian statistics, neural network regularization
// Reference: Abramowitz & Stegun §6.3

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
}

const PI: f32 = 3.14159265358979323846;
const EULER_MASCHERONI: f32 = 0.5772156649015329;

// Bernoulli numbers for asymptotic expansion: B₂=1/6, B₄=-1/30, B₆=1/42, B₈=-1/30...
// Expansion term coefficients: B₂ₖ/(2k) for k=1,2,3,...
// k=1: 1/6 / 2 = 1/12 = 0.08333333
// k=2: -1/30 / 4 = -1/120 = -0.00833333
// k=3: 1/42 / 6 = 1/252 = 0.00396825
// k=4: -1/30 / 8 = -1/240 = -0.00416667
// k=5: 5/66 / 10 = 1/132 = 0.00757576
// k=6: -691/2730 / 12 = ...
const B2_COEF: f32 = 0.08333333333333333;  // 1/12
const B4_COEF: f32 = -0.00833333333333333; // -1/120
const B6_COEF: f32 = 0.00396825396825397;  // 1/252
const B8_COEF: f32 = -0.00416666666666667; // -1/240
const B10_COEF: f32 = 0.00757575757575758; // 1/132

// Digamma for x > 6 using asymptotic expansion
fn digamma_asymptotic(x: f32) -> f32 {
    let inv_x = 1.0 / x;
    let inv_x2 = inv_x * inv_x;

    // ψ(x) ≈ ln(x) - 1/(2x) - B₂/(2·x²) - B₄/(4·x⁴) - ...
    // = ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    var sum = log(x) - 0.5 * inv_x;

    var term = inv_x2;
    sum = sum - B2_COEF * term;  // 1/(12x²)

    term = term * inv_x2;
    sum = sum - B4_COEF * term;  // -1/(120x⁴)

    term = term * inv_x2;
    sum = sum - B6_COEF * term;  // 1/(252x⁶)

    term = term * inv_x2;
    sum = sum - B8_COEF * term;

    term = term * inv_x2;
    sum = sum - B10_COEF * term;

    return sum;
}

// Full digamma function (iterative, no recursion for WGSL)
fn digamma(x: f32) -> f32 {
    // Handle special cases
    if (x <= 0.0 && floor(x) == x) {
        // Non-positive integer: pole
        let z = x - x;
        return z / z;  // NaN
    }

    var y = x;
    var result: f32 = 0.0;

    // Reflection formula for x < 0 (applied once, iteratively)
    if (y < 0.0) {
        // ψ(x) = ψ(1-x) - π·cot(πx)
        let cot_pi_x = cos(PI * y) / sin(PI * y);
        result = result - PI * cot_pi_x;
        y = 1.0 - y;
    }

    // For small y, use recurrence to shift to larger argument
    // ψ(x) = ψ(x+1) - 1/x  =>  ψ(x+1) = ψ(x) + 1/x
    // We accumulate: ψ(x) = ψ(y) - Σ 1/(x+k) for y = x + n >= 6
    while (y < 6.0) {
        result = result - 1.0 / y;
        y = y + 1.0;
    }

    // Use asymptotic expansion for y >= 6
    return result + digamma_asymptotic(y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = digamma(input[idx]);
}
