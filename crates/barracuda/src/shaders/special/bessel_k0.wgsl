// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified Bessel function of the third kind, order 0: K0(x)
// Abramowitz & Stegun 9.8.3-9.8.6
// For 0 < x <= 2: polynomial in x² + I0(x) * ln(x/2)
// For x > 2: exp(-x)/sqrt(x) * polynomial in (2/x)
// Returns infinity for x <= 0

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

// I0 for |x| < 3.75 (used in K0 small-x region)
fn i0_small(x: f32) -> f32 {
    let t = (x / 3.75) * (x / 3.75);
    return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
}

fn bessel_k0_approx(x: f32) -> f32 {
    if (x <= 0.0) {
        return 1e30; // K0 singular at x=0, return large value for x <= 0
    }
    if (x <= 2.0) {
        // 0 < x <= 2: K0(x) = -ln(x/2)*I0(x) + P(t) where t = (x/2)²
        let z = (x * 0.5) * (x * 0.5);
        let p = -0.57721566 + z * (0.42278420 + z * (0.23069756 + z * (0.03488590 + z * (0.00262698 + z * (0.00010750 + z * 0.00000740)))));
        let i0_val = i0_small(x);
        return p - i0_val * log(x / 2.0);
    }
    // x > 2: K0(x) = exp(-x)/sqrt(x) * P(2/x)
    let t = 2.0 / x;
    let p = 1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446 + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208)))));
    return exp(-x) * p / sqrt(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }
    output[idx] = bessel_k0_approx(input[idx]);
}
