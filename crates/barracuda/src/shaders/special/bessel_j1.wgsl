// SPDX-License-Identifier: AGPL-3.0-or-later
// Bessel function of the first kind, order 1: J1(x)
// Uses rational polynomial approximation (Abramowitz & Stegun 9.4.1-9.4.3)
// For |x| < 8: x * P(x²)/Q(x²)
// For |x| >= 8: asymptotic form with different phase

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

fn bessel_j1_approx(x: f32) -> f32 {
    let ax = abs(x);
    if (ax >= 8.0) {
        // Asymptotic form for |x| >= 8
        // J1(x) ≈ sqrt(2/(πx)) * [P1(8/x)*cos(x - 3π/4) - Q1(8/x)*sin(x - 3π/4)]
        let z = 8.0 / ax;
        let z2 = z * z;
        let z4 = z2 * z2;
        let p1 = 1.0 + z2 * (0.0000000000703235 + z2 * (-0.0000000033528824))
                 + z4 * (0.000000134239019 - z2 * 0.000009219976742);
        let q1 = z * (0.0000000004874090 + z2 * (-0.000000024761993))
                 + z4 * (0.000001072915446 - z2 * 0.000075272620851);
        let pp = 0.7978845608028654; // sqrt(2/π)
        let inv_sqrt_x = pp / sqrt(ax);
        let xx = ax - 2.3561944901923449; // 3π/4
        let r = inv_sqrt_x * (p1 * cos(xx) - q1 * sin(xx));
        return select(r, -r, x < 0.0);
    }
    // For |x| < 8: J1(x) = x * P(z)/Q(z) where z = x²
    let z = x * x;
    let p = 1.0 + z * (-0.0001831052 + z * (-0.0003516492 + z * (0.000002451757 - z * 0.000000052588)));
    let q = 1.0 + z * (0.04687499895 + z * (-0.0002002692 + z * (0.000001117864 - z * 0.000000020128)));
    let r = x * (p / q);
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }
    output[idx] = bessel_j1_approx(input[idx]);
}
