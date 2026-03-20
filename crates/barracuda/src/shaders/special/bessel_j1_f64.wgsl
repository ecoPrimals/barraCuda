// SPDX-License-Identifier: AGPL-3.0-or-later
// Bessel function of the first kind, order 1: J1(x) — f64 precision
// Uses rational polynomial approximation (Abramowitz & Stegun 9.4.4-9.4.6)
// For |x| < 8: x * P(x²)/Q(x²)
// For |x| >= 8: asymptotic form with different phase
//
// **f64 precision**: Critical for scientific computing with high argument values

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const SQRT_2_OVER_PI: f64 = 0.7978845608028654;  // sqrt(2/π)
const THREE_PI_OVER_4: f64 = 2.3561944901923449;  // 3π/4

fn bessel_j1_approx(x: f64) -> f64 {
    let ax = abs(x);
    if (ax >= 8.0) {
        // Asymptotic form for |x| >= 8
        // J1(x) ≈ sqrt(2/(πx)) * [P1(8/x)*cos(x - 3π/4) - Q1(8/x)*sin(x - 3π/4)]
        let z = 8.0 / ax;
        let z2 = z * z;
        let z4 = z2 * z2;
        let z6 = z4 * z2;
        let z8 = z4 * z4;
        
        // P1 coefficients for f64 precision
        let p1 = 1.0 
            + 1.83105e-3 * z2 
            - 3.516396496e-4 * z4
            + 2.457520174e-5 * z6
            - 2.40337019e-6 * z8;
        
        // Q1 coefficients
        let q1 = 4.687499995e-2 * z
            - 2.002690873e-4 * z * z2
            + 8.449199096e-6 * z * z4
            - 8.8228987e-7 * z * z6
            + 1.057874120e-7 * z * z8;
        
        let inv_sqrt_x = SQRT_2_OVER_PI / sqrt(ax);
        let xx = ax - THREE_PI_OVER_4;
        let r = inv_sqrt_x * (p1 * cos(xx) - q1 * sin(xx));
        return select(r, -r, x < 0.0);
    }
    
    // For |x| < 8: J1(x) = x * P(z)/Q(z) where z = x²
    let z = x * x;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    
    // Numerator
    let p = 72362614232.0 
        - 7895059235.0 * z 
        + 242396853.1 * z2 
        - 2972611.439 * z3 
        + 15704.48260 * z4 
        - 30.16036606 * z2 * z3;
    
    // Denominator
    let q = 144725228442.0 
        + 2300535178.0 * z 
        + 18583304.74 * z2 
        + 99447.43394 * z3 
        + 376.9991397 * z4 
        + 1.0 * z2 * z3;
    
    return x * (p / q);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }
    output[idx] = bessel_j1_approx(input[idx]);
}
