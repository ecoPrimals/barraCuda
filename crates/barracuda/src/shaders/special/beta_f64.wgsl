// SPDX-License-Identifier: AGPL-3.0-only
// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) — f64 precision
// For numerical stability: B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
//
// Input: pairs [a₀, b₀, a₁, b₁, ...] interleaved (f64)
// Output: [B(a₀,b₀), B(a₁,b₁), ...]
//
// Applications: Beta distributions, Bayesian statistics, binomial coefficients
// Reference: Abramowitz & Stegun §6.2
//
// Note: Requires GPU f64 support including log/exp operations.
// Many GPUs (especially AMD) may not support f64 transcendentals.

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,   // Number of output elements (pairs)
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

// Beta function B(a,b) via log-gamma for numerical stability
fn beta_f64(a: f64, b: f64) -> f64 {
    if (a <= f64(0.0) || b <= f64(0.0)) {
        let z = a - a;
        return z / z;  // NaN for non-positive arguments
    }
    // B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    let log_beta = lgamma_f64(a) + lgamma_f64(b) - lgamma_f64(a + b);
    return exp(log_beta);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    // Input is interleaved pairs: [a₀, b₀, a₁, b₁, ...]
    let a = input[idx * 2u];
    let b = input[idx * 2u + 1u];

    let result = beta_f64(a, b);
    output[idx] = result;
}
