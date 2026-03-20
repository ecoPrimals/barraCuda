// SPDX-License-Identifier: AGPL-3.0-or-later
// ============================================================================
// math_f64_special.wgsl — Special functions (gamma, erf, bessel)
// ============================================================================
//
// Split from math_f64.wgsl to keep files under 1000 lines.
// Dependencies: exp_f64, pow_f64, sin_f64, cos_f64 from math_f64.wgsl
// Concatenated at compile time by ShaderTemplate::math_f64_preamble().

// ============================================================================
// GAMMA FUNCTION (Lanczos approximation)
// ============================================================================

/// Lanczos core: Gamma(x) for x >= 0.5 via Lanczos approximation (g=7, n=9)
/// Split out so gamma_f64 can call it iteratively (WGSL forbids recursion).
/// PRECISION FIX (Feb 16 2026): Uses (zero + literal) pattern for full f64 precision.
fn lanczos_core_f64(x: f64) -> f64 {
    let zero = x - x;
    let one = zero + 1.0;
    let half = zero + 0.5;
    let g = zero + 7.0;
    
    // Lanczos coefficients via (zero + literal) for full f64 precision
    let c0 = zero + 0.99999999999980993;
    let c1 = zero + 676.5203681218851;
    let c2 = zero - 1259.1392167224028;
    let c3 = zero + 771.32342877765313;
    let c4 = zero - 176.61502916214059;
    let c5 = zero + 12.507343278686905;
    let c6 = zero - 0.13857109526572012;
    let c7 = zero + 9.9843695780195716e-6;
    let c8 = zero + 1.5056327351493116e-7;

    let z = x - one;

    var sum = c0;
    sum = sum + c1 / (z + one);
    sum = sum + c2 / (z + (zero + 2.0));
    sum = sum + c3 / (z + (zero + 3.0));
    sum = sum + c4 / (z + (zero + 4.0));
    sum = sum + c5 / (z + (zero + 5.0));
    sum = sum + c6 / (z + (zero + 6.0));
    sum = sum + c7 / (z + (zero + 7.0));
    sum = sum + c8 / (z + (zero + 8.0));

    let t = z + g + half;
    let sqrt_2pi = zero + 2.5066282746310005;

    return sqrt_2pi * pow_f64(t, z + half) * exp_f64(-t) * sum;
}

/// Gamma function using Lanczos approximation (non-recursive)
/// Accurate to ~15 digits for positive real arguments.
/// Reflection formula for x < 0.5 inlined to avoid WGSL recursion ban.
/// PRECISION FIX (Feb 16 2026): Uses (zero + literal) pattern for full f64 precision.
fn gamma_f64(x: f64) -> f64 {
    let zero = x - x;
    let half = zero + 0.5;
    let one = zero + 1.0;
    let pi = zero + 3.141592653589793;

    if (x < half) {
        // Reflection: Gamma(x) = pi / (sin(pi*x) * Gamma(1-x))
        // Since 1-x >= 0.5, lanczos_core handles it directly.
        let sin_pix = sin_f64(pi * x);
        let tiny = zero + 1e-15;
        if (abs(sin_pix) < tiny) {
            let big = zero + 1e38;
            return big * big;  // Pole (~1e76, large enough)
        }
        return pi / (sin_pix * lanczos_core_f64(one - x));
    }

    return lanczos_core_f64(x);
}

// ============================================================================
// ERROR FUNCTION (erf)
// ============================================================================

/// Error function using Abramowitz & Stegun approximation
/// PRECISION FIX (Feb 16 2026): Uses (zero + literal) pattern for full f64 precision.
fn erf_f64(x: f64) -> f64 {
    let zero = x - x;
    let one = zero + 1.0;
    
    // Constants via (zero + literal) for full f64 precision
    let a1 = zero + 0.254829592;
    let a2 = zero - 0.284496736;
    let a3 = zero + 1.421413741;
    let a4 = zero - 1.453152027;
    let a5 = zero + 1.061405429;
    let p = zero + 0.3275911;
    
    let sign = sign(x);
    let ax = abs(x);
    
    let t = one / (one + p * ax);
    let t2 = t * t;
    let t3 = t2 * t;
    let t4 = t3 * t;
    let t5 = t4 * t;
    
    let y = one - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * exp_f64(-ax * ax);
    
    return sign * y;
}

// ============================================================================
// F64 ENCODING/DECODING HELPERS
// ============================================================================
//
// NAGA LIMITATION (wgpu 0.19 / Naga 0.14):
// bitcast<f64>(vec2<u32>) is NOT supported, even though WGSL spec allows it.
// Workaround: Use these helper functions or integer-ratio encoding.
//
// When Naga catches up to WGSL spec, these can be replaced with bitcast.

/// Encode f64 as vec2<u32> for storage buffers that require u32 types.
/// Uses the IEEE 754 bit representation split across lo (bits 0-31) and hi (bits 32-63).
/// NOTE: This requires native bitcast support which Naga 0.14 lacks. 
/// For now, prefer using native f64 arrays or the pattern: f64(num) / f64(den).
// fn encode_f64(x: f64) -> vec2<u32> {
//     return bitcast<vec2<u32>>(x);  // NOT SUPPORTED IN NAGA 0.14
// }

/// Decode vec2<u32> back to f64.
/// NOTE: Not supported in Naga 0.14. See encode_f64 comments.
// fn decode_f64(v: vec2<u32>) -> f64 {
//     return bitcast<f64>(v);  // NOT SUPPORTED IN NAGA 0.14
// }

/// Alternative: Encode f64 as ratio of two i32 values (numerator / denominator).
/// Useful for passing constants from CPU to GPU with full precision.
/// Example: To pass 0.333... → encode as (1, 3) → GPU computes f64(1) / f64(3)
fn decode_f64_ratio(num: i32, den: i32, x_ref: f64) -> f64 {
    let zero = x_ref - x_ref;
    return (zero + f64(num)) / (zero + f64(den));
}

// ============================================================================
// BESSEL FUNCTIONS (J0, J1)
// ============================================================================

/// Bessel function J0 using polynomial approximation
/// PRECISION FIX (Feb 16 2026): Uses (zero + literal) pattern for full f64 precision.
fn bessel_j0_f64(x: f64) -> f64 {
    let zero = x - x;
    let one = zero + 1.0;
    let ax = abs(x);
    let eight = zero + 8.0;
    
    if (ax < eight) {
        let y = x * x;
        let num = (zero + 57568490574.0) + y * (
            (zero - 13362590354.0) + y * (
            (zero + 651619640.7) + y * (
            (zero - 11214424.18) + y * (
            (zero + 77392.33017) + y * (zero - 184.9052456)))));
        let den = (zero + 57568490411.0) + y * (
            (zero + 1029532985.0) + y * (
            (zero + 9494680.718) + y * (
            (zero + 59272.64853) + y * (
            (zero + 267.8532712) + y))));
        return num / den;
    } else {
        let z = eight / ax;
        let y = z * z;
        let xx = ax - (zero + 0.785398164);
        let p0 = one + y * ((zero - 0.0010986286270000001) + y * (zero + 0.000027345104070000003));
        let q0 = (zero - 0.01562499995) + y * ((zero + 0.0001430488765) + y * (zero - 0.0000069111476510000005));
        return sqrt((zero + 0.636619772) / ax) * (cos_f64(xx) * p0 - z * sin_f64(xx) * q0);
    }
}

// ============================================================================
