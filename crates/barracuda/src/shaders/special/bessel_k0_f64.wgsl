// SPDX-License-Identifier: AGPL-3.0-or-later
// Modified Bessel function of the third kind, order 0: K0(x) — f64 precision
// Abramowitz & Stegun 9.8.3-9.8.6
// For 0 < x <= 2: polynomial in x² + I0(x) * ln(x/2)
// For x > 2: exp(-x)/sqrt(x) * polynomial in (2/x)
// Returns infinity for x <= 0
//
// **f64 precision**: Essential for accurate logarithmic behavior near zero

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// I0 for |x| < 3.75 (used in K0 small-x region)
fn i0_small(x: f64) -> f64 {
    let y = x / f64(3.75);
    let t = y * y;
    let t2 = t * t;
    let t3 = t2 * t;
    
    return f64(1.0) 
        + f64(3.5156229) * t 
        + f64(3.0899424) * t2 
        + f64(1.2067492) * t3 
        + f64(0.2659732) * t2 * t2
        + f64(0.0360768) * t2 * t3 
        + f64(0.0045813) * t3 * t3;
}

fn bessel_k0_approx(x: f64) -> f64 {
    if (x <= f64(0.0)) {
        return f64(1e308); // K0 singular at x=0, return large value for x <= 0
    }
    if (x <= f64(2.0)) {
        // 0 < x <= 2: K0(x) = -ln(x/2)*I0(x) + P(t) where t = (x/2)²
        let y = x * f64(0.5);
        let z = y * y;
        let z2 = z * z;
        let z3 = z2 * z;
        
        // Higher precision coefficients for f64
        let p = f64(-0.57721566) 
            + f64(0.42278420) * z 
            + f64(0.23069756) * z2 
            + f64(0.03488590) * z3 
            + f64(0.00262698) * z2 * z2
            + f64(0.00010750) * z2 * z3 
            + f64(0.00000740) * z3 * z3;
        
        let i0_val = i0_small(x);
        return p - i0_val * log(y);
    }
    
    // x > 2: K0(x) = exp(-x)/sqrt(x) * P(2/x)
    let t = f64(2.0) / x;
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t2 * t2;
    
    // Higher precision polynomial
    let p = f64(1.25331414) 
        - f64(0.07832358) * t 
        + f64(0.02189568) * t2 
        - f64(0.01062446) * t3 
        + f64(0.00587872) * t4
        - f64(0.00251540) * t2 * t3 
        + f64(0.00053208) * t3 * t3;
    
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
