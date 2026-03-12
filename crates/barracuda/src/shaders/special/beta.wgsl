// SPDX-License-Identifier: AGPL-3.0-only
// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
// For numerical stability: B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
//
// Input: pairs [a₀, b₀, a₁, b₁, ...] interleaved
// Output: [B(a₀,b₀), B(a₁,b₁), ...]
//
// Applications: Beta distributions, Bayesian statistics, binomial coefficients
// Reference: Abramowitz & Stegun §6.2

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,   // Number of output elements (pairs)
}

// Lanczos approximation for lgamma (embedded for WGSL module independence)
fn lgamma_lanczos(x: f32) -> f32 {
    let g = 7.0;
    let x_shifted = x - 1.0;
    var sum = 0.99999999999980993;
    sum += 676.5203681218851 / (x_shifted + 1.0);
    sum += -1259.1392167224028 / (x_shifted + 2.0);
    sum += 771.32342877765313 / (x_shifted + 3.0);
    sum += -176.61502916214059 / (x_shifted + 4.0);
    sum += 12.507343278686905 / (x_shifted + 5.0);
    sum += -0.13857109526572012 / (x_shifted + 6.0);
    sum += 9.9843695780195716e-6 / (x_shifted + 7.0);
    sum += 1.5056327351493116e-7 / (x_shifted + 8.0);
    let t = x_shifted + g + 0.5;
    let sqrt_2pi = 2.5066282746310002;
    return log(sqrt_2pi) + log(sum) + (x_shifted + 0.5) * log(t) - t;
}

fn lgamma_f32(x: f32) -> f32 {
    if (x <= 0.0) {
        let z = x - x;
        return z / z;  // NaN
    }
    if (x < 0.5) {
        let pi = 3.14159265358979323846;
        return log(pi / sin(pi * x)) - lgamma_lanczos(1.0 - x);
    }
    return lgamma_lanczos(x);
}

// Beta function B(a,b) via log-gamma for numerical stability
fn beta(a: f32, b: f32) -> f32 {
    if (a <= 0.0 || b <= 0.0) {
        let z = a - a;
        return z / z;  // NaN for non-positive arguments
    }
    // B(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    let log_beta = lgamma_f32(a) + lgamma_f32(b) - lgamma_f32(a + b);
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
    
    output[idx] = beta(a, b);
}
