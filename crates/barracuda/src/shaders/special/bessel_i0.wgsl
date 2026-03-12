// SPDX-License-Identifier: AGPL-3.0-only
// Modified Bessel function of the first kind, order 0: I0(x)
// Abramowitz & Stegun 9.8.1-9.8.2
// For |x| < 3.75: polynomial approximation in (x/3.75)²
// For |x| >= 3.75: exp(x)/sqrt(x) * polynomial in (3.75/x)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

fn bessel_i0_approx(x: f32) -> f32 {
    let ax = abs(x);
    if (ax < 3.75) {
        // Polynomial in t = (x/3.75)², A&S 9.8.1
        let t = (x / 3.75) * (x / 3.75);
        let p = 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492 + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
        return p;
    }
    // |x| >= 3.75: I0(x) = exp(x)/sqrt(x) * P(3.75/x)
    let t = 3.75 / ax;
    let p = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
    return exp(ax) * p / sqrt(ax);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }
    output[idx] = bessel_i0_approx(input[idx]);
}
