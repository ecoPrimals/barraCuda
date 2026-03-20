// SPDX-License-Identifier: AGPL-3.0-or-later
// Digamma function ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x) — f64 precision
// Uses reflection + recurrence + asymptotic expansion
//
// For x < 0: ψ(x) = ψ(1-x) - π·cot(πx)  [reflection formula]
// For x > 0, x < 6: use recurrence ψ(x+1) = ψ(x) + 1/x
// For x ≥ 6: asymptotic expansion ψ(x) ≈ ln(x) - 1/(2x) - Σ B₂ₖ/(2k·x²ᵏ)
//
// Applications: Fisher information, Bayesian statistics, neural network regularization
// Reference: Abramowitz & Stegun §6.3
//
// Note: Requires GPU f64 support including log/sin/cos operations.

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
}

const PI_F64: f64 = 3.14159265358979323846264338327950288;

// Bernoulli number coefficients B₂ₖ/(2k) for k=1,2,3,...
const B2_COEF_F64: f64 = 0.08333333333333333333;  // 1/12
const B4_COEF_F64: f64 = -0.00833333333333333333; // -1/120
const B6_COEF_F64: f64 = 0.00396825396825396825;  // 1/252
const B8_COEF_F64: f64 = -0.00416666666666666667; // -1/240
const B10_COEF_F64: f64 = 0.00757575757575757576; // 1/132
const B12_COEF_F64: f64 = -0.02109279609279609280; // -691/32760
const B14_COEF_F64: f64 = 0.08333333333333333333;  // 1/12

// Digamma for x > 6 using asymptotic expansion
fn digamma_asymptotic_f64(x: f64) -> f64 {
    let inv_x = f64(1.0) / x;
    let inv_x2 = inv_x * inv_x;

    var sum = log(x) - f64(0.5) * inv_x;

    var term = inv_x2;
    sum = sum - B2_COEF_F64 * term;

    term = term * inv_x2;
    sum = sum - B4_COEF_F64 * term;

    term = term * inv_x2;
    sum = sum - B6_COEF_F64 * term;

    term = term * inv_x2;
    sum = sum - B8_COEF_F64 * term;

    term = term * inv_x2;
    sum = sum - B10_COEF_F64 * term;

    term = term * inv_x2;
    sum = sum - B12_COEF_F64 * term;

    return sum;
}

// Full digamma function
fn digamma_f64(x: f64) -> f64 {
    // Handle special cases: non-positive integers
    if (x <= f64(0.0) && floor(x) == x) {
        let z = x - x;
        return z / z;  // NaN
    }

    var y = x;
    var result: f64 = f64(0.0);

    // Reflection formula for x < 0
    if (y < f64(0.0)) {
        let cot_pi_y = cos(PI_F64 * y) / sin(PI_F64 * y);
        result = result - PI_F64 * cot_pi_y;
        y = f64(1.0) - y;
    }

    // Use recurrence to shift to larger argument
    while (y < f64(6.0)) {
        result = result - f64(1.0) / y;
        y = y + f64(1.0);
    }

    // Use asymptotic expansion for y >= 6
    return result + digamma_asymptotic_f64(y);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    let x = input[idx];
    let result = digamma_f64(x);
    output[idx] = result;
}
