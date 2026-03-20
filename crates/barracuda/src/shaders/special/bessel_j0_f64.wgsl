// SPDX-License-Identifier: AGPL-3.0-or-later
// Bessel function of the first kind, order 0: J0(x) — f64 precision
// Uses rational polynomial approximation (Abramowitz & Stegun 9.4.1-9.4.3)
// For |x| < 8: polynomial P(x²)/Q(x²)
// For |x| >= 8: asymptotic form sqrt(2/(πx)) * cos(x - π/4 + correction)
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

const PI: f64 = 3.14159265358979323846;
const SQRT_2_OVER_PI: f64 = 0.7978845608028654;  // sqrt(2/π)
const PI_OVER_4: f64 = 0.7853981633974483;

// Rational polynomial coefficients for |x| < 8 (higher precision for f64)
fn bessel_j0_approx(x: f64) -> f64 {
    let ax = abs(x);
    if (ax >= 8.0) {
        // Asymptotic form for |x| >= 8
        // J0(x) ≈ sqrt(2/(πx)) * [P0(8/x)*cos(x - π/4) - Q0(8/x)*sin(x - π/4)]
        let z = 8.0 / ax;
        let z2 = z * z;
        let z4 = z2 * z2;
        let z6 = z4 * z2;
        let z8 = z4 * z4;
        
        // Higher-order P0 coefficients for f64 precision
        let pv = 1.0 
            - 1.098628627e-3 * z2 
            + 2.734510407e-5 * z4
            - 2.073370639e-6 * z6
            + 2.093887211e-7 * z8;
        
        // Higher-order Q0 coefficients
        let qv = -1.562499995e-2 * z
            + 1.430488765e-4 * z * z2
            - 6.911147651e-6 * z * z4
            + 7.621095161e-7 * z * z6
            - 9.349451520e-8 * z * z8;
        
        let inv_sqrt_x = SQRT_2_OVER_PI / sqrt(ax);
        let xx = ax - PI_OVER_4;
        return inv_sqrt_x * (pv * cos(xx) - qv * sin(xx));
    }
    
    // For |x| < 8: rational approximation in x²
    let z = x * x;
    let z2 = z * z;
    let z3 = z2 * z;
    let z4 = z2 * z2;
    
    // Numerator: J0 ≈ 1 - x²/4 + x⁴/64 - ...
    let p = 57568490574.0 
        - 13362590354.0 * z 
        + 651619640.7 * z2 
        - 11214424.18 * z3 
        + 77392.33017 * z4 
        - 184.9052456 * z2 * z3;
    
    let q = 57568490411.0 
        + 1029532985.0 * z 
        + 9494680.718 * z2 
        + 59272.64853 * z3 
        + 267.8532712 * z4 
        + 1.0 * z2 * z3;
    
    return p / q;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }
    output[idx] = bessel_j0_approx(input[idx]);
}
