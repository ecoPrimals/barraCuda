// SPDX-License-Identifier: AGPL-3.0-only
// Complementary error function (erfc) operation (f64 canonical)
// erfc(x) = 1 - erf(x)
// Approximation using Abramowitz and Stegun formula

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

// Abramowitz and Stegun approximation for erf
fn erf_approx(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp_f64(-abs_x * abs_x);

    return sign * y;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }

    let x = input[idx];

    // erfc(x) = 1 - erf(x)
    output[idx] = 1.0 - erf_approx(x);
}
