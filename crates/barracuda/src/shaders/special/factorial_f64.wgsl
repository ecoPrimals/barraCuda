// SPDX-License-Identifier: AGPL-3.0-or-later
// Factorial n! for f64 input
//
// For non-negative integer n < 21: exact lookup table (integer factorials 0!..20!).
// For n >= 21 or non-integer: Γ(n+1) = exp(lgamma(n+1)) via Lanczos approximation.
//
// Input: [n₀, n₁, n₂, ...] (as vec2<u32> for f64, one value per element)
// Output: [n₀!, n₁!, n₂!, ...] (as vec2<u32> for f64)
//
// Applications: Combinatorics, binomial coefficients, probability (Poisson, etc.),
// Taylor series, permutations and combinations.
// Reference: Γ(n+1) = n! for non-negative integers; Abramowitz & Stegun §6.1
//
// Note: Negative integer n produces NaN. Non-integer n uses gamma continuation.
// Requires GPU f64 support including log/exp operations.

@group(0) @binding(0) var<storage, read> input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,   // Number of output elements
}

// Lookup exact integer factorials 0! through 20!
fn factorial_lookup(k: i32) -> f64 {
    switch k {
        case 0: { return f64(1.0); }
        case 1: { return f64(1.0); }
        case 2: { return f64(2.0); }
        case 3: { return f64(6.0); }
        case 4: { return f64(24.0); }
        case 5: { return f64(120.0); }
        case 6: { return f64(720.0); }
        case 7: { return f64(5040.0); }
        case 8: { return f64(40320.0); }
        case 9: { return f64(362880.0); }
        case 10: { return f64(3628800.0); }
        case 11: { return f64(39916800.0); }
        case 12: { return f64(479001600.0); }
        case 13: { return f64(6227020800.0); }
        case 14: { return f64(87178291200.0); }
        case 15: { return f64(1307674368000.0); }
        case 16: { return f64(20922789888000.0); }
        case 17: { return f64(355687428096000.0); }
        case 18: { return f64(6402373705728000.0); }
        case 19: { return f64(121645100408832000.0); }
        case 20: { return f64(2432902008176640000.0); }
        default: { return f64(0.0); }
    }
}

// Constants for f64
const PI_F64: f64 = 3.14159265358979323846264338327950288;
const SQRT_2PI_F64: f64 = 2.5066282746310005024157652848110452;

// Lanczos approximation for lgamma (g=7)
fn lgamma_lanczos_f64(x: f64) -> f64 {
    let g = f64(7.0);
    let x_shifted = x - f64(1.0);
    var sum: f64 = f64(0.99999999999980993);
    sum = sum + f64(676.5203681218851) / (x_shifted + f64(1.0));
    sum = sum + f64(-1259.1392167224028) / (x_shifted + f64(2.0));
    sum = sum + f64(771.32342877765313) / (x_shifted + f64(3.0));
    sum = sum + f64(-176.61502916214059) / (x_shifted + f64(4.0));
    sum = sum + f64(12.507343278686905) / (x_shifted + f64(5.0));
    sum = sum + f64(-0.13857109526572012) / (x_shifted + f64(6.0));
    sum = sum + f64(9.9843695780195716e-6) / (x_shifted + f64(7.0));
    sum = sum + f64(1.5056327351493116e-7) / (x_shifted + f64(8.0));
    let t = x_shifted + g + f64(0.5);
    return log(SQRT_2PI_F64) + log(sum) + (x_shifted + f64(0.5)) * log(t) - t;
}

fn lgamma_f64(x: f64) -> f64 {
    if (x <= f64(0.0)) {
        let z = x - x;
        return z / z;  // NaN
    }
    if (x < f64(0.5)) {
        return log(PI_F64 / sin(PI_F64 * x)) - lgamma_lanczos_f64(f64(1.0) - x);
    }
    return lgamma_lanczos_f64(x);
}

fn factorial_f64(n: f64) -> f64 {
    if (n < f64(0.0)) {
        let z = n - n;
        return z / z;  // NaN for negative
    }
    if (n < f64(21.0)) {
        let k = i32(round(n));
        if (abs(n - f64(k)) < f64(1e-10) && k >= 0 && k <= 20) {
            return factorial_lookup(k);
        }
    }
    return exp(lgamma_f64(n + f64(1.0)));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let n = bitcast<f64>(input[idx]);
    let result = factorial_f64(n);
    output[idx] = bitcast<vec2<u32>>(result);
}
