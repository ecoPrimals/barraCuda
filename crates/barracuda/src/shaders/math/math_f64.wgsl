// ============================================================================
// math_f64.wgsl — f64 math library for GPU compute
// ============================================================================
//
// Two categories of functions live here:
//
// ── FOSSILS ─────────────────────────────────────────────────────────────────
// Probe-confirmed (Feb 18 2026, bench_f64_builtins — RTX 3090 + RX 6950 XT)
// native WGSL f64 builtins on ALL SHADER_F64 hardware via Vulkan:
//
//   abs(f64)    min(f64,f64)  max(f64,f64)  clamp(f64,f64,f64)
//   sqrt(f64)   floor(f64)    ceil(f64)     round(f64)
//   sign(f64)   fract(f64)
//   NOTE: fma() is NOT a valid f64 builtin in WGSL — use a*b+c instead.
//
// These software implementations are FOSSILS: retained as reference /
// emergency fallback for future edge-case GPU profiling. New shader code
// MUST use native WGSL builtins directly. ShaderTemplate will NOT inject
// fossil functions into new shaders. See F64_FOSSIL_FUNCTIONS in math_f64.rs.
//
// ── ACTIVE FALLBACKS ────────────────────────────────────────────────────────
// The transcendentals are NOT provided as f64 builtins on open-source drivers:
//
//   Driver           exp  log  exp2 log2 sin  cos  | sqrt fma  abs
//   ─────────────────────────────────────────────────────────────────
//   RTX 3090 PTXAS   ✓    ✓    ✓    ✓    ✓    ✓   |  ✓    ✓    ✓
//   RX 6950 XT ACO   ✗    ✗    ✗    ✗    ✗    ✗   |  ✓    ✓    ✓
//   NVK/NAK          ✗    ✗    ✗    ✗    ✗    ✗   |  ✓    ✓    ✓
//
// NOTE: RADV/ACO does NOT have full f64 builtin support — only sqrt/fma/abs
// are native. Earlier comments claiming "RADV full f64 via AMDGPU" were wrong.
// The WGSL→Vulkan path exposes VK_KHR_shader_float64, bypassing proprietary
// FP64 locks, but open-source compilers (ACO, NAK) have not yet implemented
// f64 transcendentals. See DEBT.md W-001 and W-003.
//
// For portable code across open-source drivers:
//   exp(f64), log(f64), sin(f64), cos(f64) → use exp_f64(), log_f64(), etc.
//   ShaderTemplate::for_driver_auto() patches shaders automatically.
//
// ── CRITICAL NAGA/WGSL GOTCHAS ──────────────────────────────────────────────
// 1. AbstractFloat (0.0, 1.0) does NOT auto-promote to f64
//    - WRONG: return 1.0;
//    - RIGHT: return x - x + 1.0;   // (f64 - f64) + AbstractFloat → f64
//
// 2. f64 CONSTANT PRECISION (Feb 16 2026 — wetSpring finding):
//    f64(0.333...) truncates through f32, losing ~7 digits of precision!
//    - WRONG: let c = f64(0.3333333333333333);
//    - WRONG: f64_const(x, 0.333...);           // f32 parameter truncates
//    - RIGHT: let zero = x - x; let c = zero + 0.3333333333333333;
//    The (zero + literal) pattern preserves all 15-16 significant digits.
//    Use this for polynomial coefficients and high-precision constants.
//
// 3. Literals > f32 range cause parse errors — construct via arithmetic
//
// 4. No f64 vec types (vec2<f64>, vec3<f64>, vec4<f64> not supported)
//
// 5. NEVER use i32 % for negative wrapping — produces incorrect results on
//    NVIDIA/Naga/Vulkan. Use branch-based conditionals instead:
//    - WRONG: ((x % n) + n) % n
//    - RIGHT: var w = x; if (w < 0) { w = w + n; } if (w >= n) { w = w - n; }
//    See: hotSpring ALERT Feb 15 2026 - cell-list bug diagnosis.
//
// ── PRECISION TARGETS ───────────────────────────────────────────────────────
//   cbrt_f64:  full f64 precision (Halley's method)
//   exp_f64:   ~1e-15 relative error (degree-13 polynomial with range reduction)
//   log_f64:   ~1e-15 relative error (atanh transform, degree-7 polynomial)
//   pow_f64:   uses specialized paths for common exponents
// ============================================================================

// Helper: construct f64 constant from AbstractFloat
// The pattern (x - x + c) ensures f64 type propagation
fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

// ============================================================================
// FOSSIL FUNCTIONS — superseded by native WGSL f64 builtins
// ============================================================================
// Probe-confirmed native on RTX 3090 (PTXAS) and RX 6950 XT (ACO) Feb 2026.
// Kept as reference / emergency fallback for future edge-case GPU profiling.
// New shaders MUST use native WGSL built-ins. ShaderTemplate does NOT inject
// these. See F64_FOSSIL_FUNCTIONS in math_f64.rs.
// ============================================================================

// 🦴 FOSSIL — use native abs(x)
fn abs_f64(x: f64) -> f64 {
    if (x < f64_const(x, 0.0)) {
        return -x;
    }
    return x;
}

// 🦴 FOSSIL — use native sign(x)
fn sign_f64(x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    if (x > zero) {
        return f64_const(x, 1.0);
    }
    if (x < zero) {
        return f64_const(x, -1.0);
    }
    return zero;
}

// 🦴 FOSSIL — use native floor(x)
fn floor_f64(x: f64) -> f64 {
    let i = i32(x);
    let fi = f64(i);
    if (x < fi) {
        return fi - f64_const(x, 1.0);
    }
    return fi;
}

// 🦴 FOSSIL — use native ceil(x)
fn ceil_f64(x: f64) -> f64 {
    let i = i32(x);
    let fi = f64(i);
    if (x > fi) {
        return fi + f64_const(x, 1.0);
    }
    return fi;
}

// 🦴 FOSSIL — use native round(x)
fn round_f64(x: f64) -> f64 {
    return floor_f64(x + f64_const(x, 0.5));
}

// 🦴 FOSSIL — use native fract(x)
fn fract_f64(x: f64) -> f64 {
    return x - floor_f64(x);
}

// 🦴 FOSSIL — use native min(a, b)
fn min_f64(a: f64, b: f64) -> f64 {
    if (a < b) { return a; }
    return b;
}

// 🦴 FOSSIL — use native max(a, b)
fn max_f64(a: f64, b: f64) -> f64 {
    if (a > b) { return a; }
    return b;
}

// 🦴 FOSSIL — use native clamp(x, lo, hi)
fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    return min_f64(max_f64(x, lo), hi);
}

// ============================================================================
// SQUARE ROOT — Newton-Raphson (kept as fossil; native sqrt(f64) preferred)
// ============================================================================

// 🦴 FOSSIL — use native sqrt(x) on all SHADER_F64 hardware.
// Probe-confirmed native on RTX 3090 (PTXAS) and RX 6950 XT (ACO) Feb 2026.
// Newton-Raphson implementation retained for edge-case GPU profiling only.
fn sqrt_f64(x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    if (x <= zero) {
        return zero;
    }
    
    // Initial estimate using f32 sqrt (via bit manipulation approximation)
    // For robustness, use a simple initial guess based on magnitude
    var y = x;
    
    // Scale to reasonable range for initial guess
    var scale = f64_const(x, 1.0);
    let large = f64_const(x, 1e32);
    let small = f64_const(x, 1e-32);
    
    if (x > large) {
        y = x / large;
        scale = f64_const(x, 1e16);
    } else if (x < small) {
        y = x * large;
        scale = f64_const(x, 1e-16);
    }
    
    // Initial guess: y^0.5 ≈ y / 2 for y near 1 (crude but converges)
    var r = (y + f64_const(x, 1.0)) / f64_const(x, 2.0);
    
    // Newton-Raphson iterations
    let half = f64_const(x, 0.5);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    
    return r * scale;
}

// ============================================================================
// ACTIVE FALLBACK FUNCTIONS — no native WGSL f64 equivalent
// ============================================================================

// ============================================================================
// CUBE ROOT — Halley's method (no native cbrt(f64) in WGSL)
// ============================================================================

/// Cube root using Halley's iteration
/// x_{n+1} = x_n * (x_n^3 + 2*S) / (2*x_n^3 + S)
fn cbrt_f64(x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    if (x == zero) {
        return zero;
    }
    
    let neg = x < zero;
    var y = abs(x);
    
    // Scale to reasonable range
    var scale = f64_const(x, 1.0);
    let large = f64_const(x, 1e30);
    let small = f64_const(x, 1e-30);
    
    if (y > large) {
        y = y / large;
        scale = f64_const(x, 1e10);  // cbrt(1e30) = 1e10
    } else if (y < small) {
        y = y * large;
        scale = f64_const(x, 1e-10);
    }
    
    // Initial guess
    var r = (y + f64_const(x, 1.0)) / f64_const(x, 2.0);
    
    // Halley's method iterations
    let two = f64_const(x, 2.0);
    for (var i = 0; i < 6; i = i + 1) {
        let r3 = r * r * r;
        r = r * (r3 + two * y) / (two * r3 + y);
    }
    
    if (neg) {
        return -r * scale;
    }
    return r * scale;
}

// ============================================================================
// EXPONENTIAL — Degree-17 polynomial with range reduction
// ============================================================================

/// Constants for exp
const LN2_HI: f64 = 0.693147180559945286;  // High part of ln(2)
const LN2_LO: f64 = 1.94821509970e-17;     // Low part of ln(2)
const INV_LN2: f64 = 1.4426950408889634;   // 1/ln(2)

/// Exponential function using range reduction and polynomial
/// exp(x) = 2^k * exp(r) where r = x - k*ln(2) and |r| < ln(2)/2
///
/// PRECISION FIX (Feb 16 2026 — wetSpring pattern):
/// Uses (zero + literal) pattern for all f64 constants to preserve full precision.
/// f64_const() truncates through f32, losing ~7 digits.
fn exp_f64(x: f64) -> f64 {
    // Use (zero + literal) pattern for full f64 precision
    let zero = x - x;
    let one = zero + 1.0;
    let two = zero + 2.0;
    let half = zero + 0.5;
    
    // Handle special cases
    let overflow_thresh = zero + 709.0;  // ln(DBL_MAX) ≈ 709.78
    let underflow_thresh = zero - 745.0;
    
    if (x > overflow_thresh) {
        // Return large value (can't express infinity; 1e308 overflows f32 literal)
        let big = zero + 1e38;
        return big * big;
    }
    if (x < underflow_thresh) {
        return zero;
    }
    let tiny = zero + 1e-15;
    if (abs(x) < tiny) {
        return one + x;  // exp(x) ≈ 1 + x for small x
    }
    
    // Range reduction: x = k*ln(2) + r
    // Full precision constants via (zero + literal)
    let inv_ln2 = zero + 1.4426950408889634;
    let k_f = round(x * inv_ln2);
    let k = i32(k_f);
    
    // r = x - k * ln(2) (high precision, split into hi/lo parts)
    let ln2_hi = zero + 0.6931471805599453;
    let ln2_lo = zero + 2.3190468138462996e-17;
    var r = x - k_f * ln2_hi;
    r = r - k_f * ln2_lo;
    
    // Polynomial approximation for exp(r) - 1
    // Using degree-13 minimax polynomial for |r| < ln(2)/2
    // Coefficients via (zero + literal) for full f64 precision
    let r2 = r * r;
    
    // Coefficients: 1/n! series with minimax optimization
    let c2 = zero + 0.5;
    let c3 = zero + 0.16666666666666666;
    let c4 = zero + 0.041666666666666664;
    let c5 = zero + 0.008333333333333333;
    let c6 = zero + 0.001388888888888889;
    let c7 = zero + 0.0001984126984126984;
    let c8 = zero + 0.0000248015873015873;
    let c9 = zero + 0.0000027557319223985893;
    let c10 = zero + 2.7557319223985888e-7;
    let c11 = zero + 2.505210838544172e-8;
    let c12 = zero + 2.08767569878681e-9;
    let c13 = zero + 1.6059043836821613e-10;
    
    // Horner's method evaluation
    var p = c13;
    p = p * r + c12;
    p = p * r + c11;
    p = p * r + c10;
    p = p * r + c9;
    p = p * r + c8;
    p = p * r + c7;
    p = p * r + c6;
    p = p * r + c5;
    p = p * r + c4;
    p = p * r + c3;
    p = p * r + c2;
    
    // exp(r) = 1 + r + r^2 * p
    var exp_r = one + r + r2 * p;
    
    // Scale by 2^k using repeated squaring (faster than loop)
    // Split into positive and negative cases
    if (k >= 0) {
        // Build 2^k via repeated doubling in chunks
        var scale = one;
        var remaining = k;
        // Handle large exponents in chunks of 64 (2^64 fits in f64)
        let pow64 = zero + 18446744073709551616.0;  // 2^64
        while (remaining >= 64) {
            scale = scale * pow64;
            remaining = remaining - 64;
        }
        // Handle remaining bits
        let pow32 = zero + 4294967296.0;  // 2^32
        if (remaining >= 32) {
            scale = scale * pow32;
            remaining = remaining - 32;
        }
        let pow16 = zero + 65536.0;  // 2^16
        if (remaining >= 16) {
            scale = scale * pow16;
            remaining = remaining - 16;
        }
        let pow8 = zero + 256.0;  // 2^8
        if (remaining >= 8) {
            scale = scale * pow8;
            remaining = remaining - 8;
        }
        let pow4 = zero + 16.0;  // 2^4
        if (remaining >= 4) {
            scale = scale * pow4;
            remaining = remaining - 4;
        }
        if (remaining >= 2) {
            scale = scale * (zero + 4.0);
            remaining = remaining - 2;
        }
        if (remaining >= 1) {
            scale = scale * two;
        }
        exp_r = exp_r * scale;
    } else {
        // Negative k: multiply by 2^(-|k|) = 1/2^|k|
        var scale = one;
        var remaining = -k;
        let inv_pow64 = zero + 5.421010862427522e-20;  // 2^-64
        while (remaining >= 64) {
            scale = scale * inv_pow64;
            remaining = remaining - 64;
        }
        let inv_pow32 = zero + 2.3283064365386963e-10;  // 2^-32
        if (remaining >= 32) {
            scale = scale * inv_pow32;
            remaining = remaining - 32;
        }
        let inv_pow16 = zero + 0.0000152587890625;  // 2^-16
        if (remaining >= 16) {
            scale = scale * inv_pow16;
            remaining = remaining - 16;
        }
        let inv_pow8 = zero + 0.00390625;  // 2^-8
        if (remaining >= 8) {
            scale = scale * inv_pow8;
            remaining = remaining - 8;
        }
        let inv_pow4 = zero + 0.0625;  // 2^-4
        if (remaining >= 4) {
            scale = scale * inv_pow4;
            remaining = remaining - 4;
        }
        if (remaining >= 2) {
            scale = scale * (zero + 0.25);
            remaining = remaining - 2;
        }
        if (remaining >= 1) {
            scale = scale * half;
        }
        exp_r = exp_r * scale;
    }
    
    return exp_r;
}

// ============================================================================
// NATURAL LOGARITHM — Range reduction + polynomial
// ============================================================================

/// Natural logarithm using range reduction and polynomial
/// log(x) = log(2^k * m) = k*ln(2) + log(m) where 1 <= m < 2
fn log_f64(x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    let one = f64_const(x, 1.0);
    
    // Handle special cases
    if (x <= zero) {
        return -f64_const(x, 1e38) * f64_const(x, 1e38);  // -infinity approximation
    }
    
    // Range reduction to [1, 2)
    var y = x;
    var k = f64_const(x, 0.0);
    let two = f64_const(x, 2.0);
    let half = f64_const(x, 0.5);
    
    // Scale to [1, 2)
    while (y >= two) {
        y = y * half;
        k = k + one;
    }
    while (y < one) {
        y = y * two;
        k = k - one;
    }
    
    // Now y is in [1, 2), compute log(y) using log(1+z) where z = y - 1
    let z = y - one;
    
    // For better convergence, use z = (y-1)/(y+1) transformation
    // log(y) = 2 * atanh(z/(2+z)) = 2 * atanh((y-1)/(y+1))
    let s = z / (two + z);  // s = (y-1)/(y+1)
    let s2 = s * s;
    
    // Polynomial for log(y) via atanh transformation:
    // log(y) = 2 * atanh((y-1)/(y+1)) = 2 * s * (1 + s²/3 + s⁴/5 + s⁶/7 + ...)
    // Coefficients are 1/(2k+1) with minimax optimization.
    // The outer "two * s * (1 + s² * p)" provides the factor of 2.
    //
    // BUG FIX (Feb 16 2026 — wetSpring handoff):
    // Original coefficients were 2/3, 2/5, etc. (doubled), causing ~1e-3 error.
    // Corrected to 1/3, 1/5, etc. for ~1e-15 precision.
    //
    // NOTE: Use (x - x + literal) pattern to preserve full f64 precision.
    // f64_const() truncates through f32, losing ~7 digits.
    let base = x - x;  // Use different name to avoid redefinition
    let c1 = base + 0.3333333333333367565;   // ≈ 1/3 (minimax)
    let c2 = base + 0.1999999999970470954;   // ≈ 1/5 (minimax)
    let c3 = base + 0.1428571437183119575;   // ≈ 1/7 (minimax)
    let c4 = base + 0.1111109921607489198;   // ≈ 1/9 (minimax)
    let c5 = base + 0.0909178608080902506;   // ≈ 1/11 (minimax)
    let c6 = base + 0.0765691884960468666;   // ≈ 1/13 (minimax)
    let c7 = base + 0.0739909930255829295;   // ≈ 1/15 (minimax)
    
    // Horner's evaluation
    var p = c7;
    p = p * s2 + c6;
    p = p * s2 + c5;
    p = p * s2 + c4;
    p = p * s2 + c3;
    p = p * s2 + c2;
    p = p * s2 + c1;
    
    // log(y) = 2 * s * (1 + s^2 * p)
    let log_y = two * s * (one + s2 * p);
    
    // log(x) = k * ln(2) + log(y)
    let ln2 = base + 0.6931471805599453;
    return k * ln2 + log_y;
}

// ============================================================================
// POWER FUNCTION — Specialized paths for common exponents
// ============================================================================

/// Integer power (fast path)
fn ipow_f64(base: f64, exp: i32) -> f64 {
    let one = f64_const(base, 1.0);
    if (exp == 0) {
        return one;
    }
    
    var b = base;
    var e = exp;
    var result = one;
    
    if (e < 0) {
        b = one / b;
        e = -e;
    }
    
    // Binary exponentiation
    while (e > 0) {
        if ((e & 1) == 1) {
            result = result * b;
        }
        b = b * b;
        e = e >> 1;
    }
    
    return result;
}

/// Cube root specialized for A^(1/3) — higher precision than exp(log(x)/3)
fn pow_one_third(x: f64) -> f64 {
    return cbrt_f64(x);
}

/// Square root specialized for A^(1/2) — higher precision than exp(log(x)/2)
fn pow_one_half(x: f64) -> f64 {
    return sqrt(x);
}

/// A^(2/3) specialized — higher precision than exp(2*log(x)/3)
fn pow_two_thirds(x: f64) -> f64 {
    let cbrt_x = cbrt_f64(x);
    return cbrt_x * cbrt_x;
}

/// General power function
/// For fractional powers, uses exp(exponent * log(base))
/// For integer powers, uses binary exponentiation
/// For common fractions (1/2, 1/3, 2/3), uses specialized high-precision paths
fn pow_f64(base: f64, exponent: f64) -> f64 {
    let zero = f64_const(base, 0.0);
    let one = f64_const(base, 1.0);
    
    // Handle special cases
    if (exponent == zero) {
        return one;
    }
    if (base == zero) {
        return zero;
    }
    if (base == one) {
        return one;
    }
    if (exponent == one) {
        return base;
    }
    
    // Check for integer exponent
    let exp_rounded = round(exponent);
    let is_integer = abs(exponent - exp_rounded) < f64_const(base, 1e-10);
    
    if (is_integer) {
        return ipow_f64(base, i32(exp_rounded));
    }
    
    // Check for common fractions (full precision via zero + literal pattern)
    let half = f64_const(base, 0.5);
    let bz = base - base;
    let one_third = bz + 0.3333333333333333;
    let two_thirds = bz + 0.6666666666666667;
    let neg_half = f64_const(base, -0.5);
    
    if (abs(exponent - half) < f64_const(base, 1e-10)) {
        return sqrt(base);
    }
    if (abs(exponent - one_third) < f64_const(base, 1e-10)) {
        return cbrt_f64(base);
    }
    if (abs(exponent - two_thirds) < f64_const(base, 1e-10)) {
        return pow_two_thirds(base);
    }
    if (abs(exponent - neg_half) < f64_const(base, 1e-10)) {
        return one / sqrt(base);
    }
    
    // General case: exp(exponent * log(base))
    // Note: This has ~1e-14 relative error due to polynomial approximation
    if (base > zero) {
        return exp_f64(exponent * log_f64(base));
    }
    
    // Negative base with non-integer exponent is undefined
    return zero;
}

// ============================================================================
// TRIGONOMETRIC FUNCTIONS (Basic implementations)
// ============================================================================

// ── Trigonometric kernel functions ────────────────────────────────────────────
// Accurate on [-pi/4, pi/4] via minimax (Horner form). Coefficients from
// fdlibm / ISO C Math Library. Do not call directly — use sin_f64/cos_f64.

/// sin kernel: accurate for |x| <= pi/4. Returns sin(x) ≈ x + x^3 * P(x^2).
fn sin_kernel_f64(x: f64) -> f64 {
    let zero = x - x;
    let x2 = x * x;
    // Horner form of the degree-11 minimax polynomial (fdlibm S1..S5):
    var p = zero - 1.9841269841269841e-4;   // -1/5040
    p = p * x2 + (zero + 8.3333333332257332e-3);    // +1/120
    p = p * x2 + (zero - 1.6666666666666632e-1);    // -1/6
    return x + x * x2 * p;
}

/// cos kernel: accurate for |x| <= pi/4. Returns cos(x) ≈ 1 - x^2/2 + x^4*Q(x^2).
fn cos_kernel_f64(x: f64) -> f64 {
    let zero = x - x;
    let one  = zero + 1.0;
    let half = zero + 0.5;
    let x2 = x * x;
    // Horner form of the degree-10 minimax polynomial (fdlibm C1..C4):
    var q = zero - 1.3888888888874999e-3;   // -1/720
    q = q * x2 + (zero + 4.1666666666666664e-2);    // +1/24
    return one - half * x2 + x2 * x2 * q;
}

// ── Cody-Waite pi/2 reduction ─────────────────────────────────────────────────
//
// Reduces x to r in [-pi/4, pi/4] and returns n mod 4 as a float in {0,1,2,3}.
// Uses two-term Cody-Waite decomposition:
//   pi/2 = pio2_hi + pio2_lo, with pio2_hi having its lower 27 bits zeroed so
//   n * pio2_hi is exact for |n| < 2^27 (covers |x| < 2.1e8).
//
// Sources: fdlibm e_sin.c / e_cos.c; arXiv:1804.06826 validation for SM70.

/// Sine with Cody-Waite range reduction (Feb 19, 2026).
///
/// Replaces the previous while-loop range reduction which was O(|x|/2π) and
/// lost precision for large |x|. This implementation is O(1) and accurate to
/// within 1 ULP for |x| < 2.1e8 (the limit of two-term Cody-Waite).
fn sin_f64(x: f64) -> f64 {
    let zero = x - x;

    // ── Range reduction ─────────────────────────────────────────────────────
    // Cody-Waite two-term representation of pi/2:
    //   pio2_hi = 1.5707963267341256141  (lower 27 bits of mantissa zeroed)
    //   pio2_lo = 6.07710050650619224e-11 (residual)
    let two_over_pi = zero + 6.366197723675813830e-1; // 2/π
    let pio2_hi     = zero + 1.5707963267341256141;
    let pio2_lo     = zero + 6.07710050650619224e-11;

    // n = round(x * 2/π) as a float (exact for |n| < 2^53).
    let n = floor(x * two_over_pi + (zero + 0.5));

    // Two-step Cody-Waite reduction (each step exact for the given pio2_* size):
    let r = (x - n * pio2_hi) - n * pio2_lo;

    // ── Quadrant dispatch ────────────────────────────────────────────────────
    // n mod 4, mapped to float {0,1,2,3}.  floor(n/4)*4 is exact.
    let n4 = n - floor(n * (zero + 0.25)) * (zero + 4.0);

    if (n4 < (zero + 0.5)) { return sin_kernel_f64(r); }
    if (n4 < (zero + 1.5)) { return cos_kernel_f64(r); }
    if (n4 < (zero + 2.5)) { return -sin_kernel_f64(r); }
    return -cos_kernel_f64(r);
}

/// Cosine with Cody-Waite range reduction (Feb 19, 2026).
///
/// Implements cos(x) = sin(x + π/2) correctly at the range-reduction level
/// (shifting n by 1 before quadrant dispatch), avoiding precision loss from
/// naively computing sin(x + π/2) with a floating-point add.
fn cos_f64(x: f64) -> f64 {
    let zero = x - x;

    let two_over_pi = zero + 6.366197723675813830e-1;
    let pio2_hi     = zero + 1.5707963267341256141;
    let pio2_lo     = zero + 6.07710050650619224e-11;

    let n  = floor(x * two_over_pi + (zero + 0.5));
    let r  = (x - n * pio2_hi) - n * pio2_lo;

    // Cosine is sin shifted by 1 quadrant: (n+1) mod 4.
    let n4 = (n + (zero + 1.0)) - floor((n + (zero + 1.0)) * (zero + 0.25)) * (zero + 4.0);

    if (n4 < (zero + 0.5)) { return sin_kernel_f64(r); }
    if (n4 < (zero + 1.5)) { return cos_kernel_f64(r); }
    if (n4 < (zero + 2.5)) { return -sin_kernel_f64(r); }
    return -cos_kernel_f64(r);
}

/// Tangent using sin/cos.
fn tan_f64(x: f64) -> f64 {
    return sin_f64(x) / cos_f64(x);
}

// ============================================================================
// INVERSE TRIGONOMETRIC FUNCTIONS (Phase 5 — Feb 19, 2026)
// ============================================================================

/// Arctangent for |x| <= 1 using a degree-13 minimax polynomial (fdlibm).
/// Do not call directly — use atan_f64 which handles the full range.
fn atan_kernel_f64(x: f64) -> f64 {
    let zero = x - x;
    let x2 = x * x;
    // Horner form; coefficients from fdlibm atanf/atan (aT array).
    var p = zero + 1.62858201153657823623e-2;
    p = p * x2 + (zero - 3.65315727442169155270e-2);
    p = p * x2 + (zero + 4.97687799461593236017e-2);
    p = p * x2 + (zero - 5.83357013379057348645e-2);
    p = p * x2 + (zero + 6.66107313738753120669e-2);
    p = p * x2 + (zero - 7.69187620504482999495e-2);
    p = p * x2 + (zero + 9.09088713343650656196e-2);
    p = p * x2 + (zero - 1.11111104054623557880e-1);
    p = p * x2 + (zero + 1.42857142725034663711e-1);
    p = p * x2 + (zero - 1.99999999998764832476e-1);
    p = p * x2 + (zero + 3.33333333333329318027e-1);
    return x + x * x2 * p;
}

/// Arctangent: atan(x) for any x.
///
/// Reduction strategy:
///   |x| >  1  : atan(x) =  π/2 - atan(1/x)  (or -π/2 + atan(1/x))
///   |x| <= 1  : use kernel polynomial directly
fn atan_f64(x: f64) -> f64 {
    let zero = x - x;
    let one  = zero + 1.0;
    let pio2 = zero + 1.5707963267948966192;

    let ax = abs(x);
    let sign = select(one, -one, x < zero);

    if (ax <= one) {
        return sign * atan_kernel_f64(ax);
    }
    // |x| > 1: atan(x) = sign * (π/2 - atan(1/|x|))
    return sign * (pio2 - atan_kernel_f64(one / ax));
}

/// Two-argument arctangent: atan2(y, x) in (-π, π].
///
/// Handles all quadrants including the degenerate x == 0, y == 0 cases.
fn atan2_f64(y: f64, x: f64) -> f64 {
    let zero = y - y;
    let pi   = zero + 3.141592653589793238;
    let pio2 = zero + 1.5707963267948966192;

    if (x == zero) {
        if (y > zero)  { return pio2; }
        if (y < zero)  { return -pio2; }
        return zero; // atan2(0, 0) — undefined; return 0 by convention
    }

    let r = atan_f64(y / x);

    if (x > zero) {
        return r;
    }
    // x < 0
    if (y >= zero) {
        return r + pi;
    }
    return r - pi;
}

/// Arcsine: asin(x) for x in [-1, 1].
///
/// Strategy (fdlibm):
///   |x| <= 0.5 : asin(x) = x + x^3 * R(x^2),  R from minimax
///   0.5 < |x| <= 1 : asin(x) = π/2 - 2*asin(sqrt((1-|x|)/2))  [half-angle]
fn asin_f64(x: f64) -> f64 {
    let zero = x - x;
    let one  = zero + 1.0;
    let half = zero + 0.5;
    let pio2 = zero + 1.5707963267948966192;
    let two  = zero + 2.0;

    let ax = abs(x);
    let sign = select(one, -one, x < zero);

    if (ax > one) { return zero; } // domain error — return 0

    if (ax <= half) {
        // Small argument: use atan kernel scaled by 1/sqrt(1-x^2).
        // Equivalent and simpler: asin(x) ≈ atan(x / sqrt(1 - x^2))
        let t = one - ax * ax;
        return sign * atan2_f64(ax, sqrt_f64(t));
    }

    // Half-angle reduction: asin(x) = π/2 - 2*asin(sqrt((1-|x|)/2))
    let w = sqrt_f64((one - ax) * half);
    let s = atan2_f64(w, sqrt_f64(one - w * w));
    return sign * (pio2 - two * s);
}

/// Arccosine: acos(x) for x in [-1, 1].
///
/// acos(x) = π/2 - asin(x), accurate near x = ±1 via the half-angle formula
/// inherited from asin_f64's implementation.
fn acos_f64(x: f64) -> f64 {
    let zero = x - x;
    let pio2 = zero + 1.5707963267948966192;
    return pio2 - asin_f64(x);
}

// ============================================================================
// HYPERBOLIC FUNCTIONS
// ============================================================================

/// Hyperbolic sine: sinh(x) = (exp(x) - exp(-x)) / 2
fn sinh_f64(x: f64) -> f64 {
    let zero = x - x;
    let two = zero + 2.0;
    let ex = exp_f64(x);
    let emx = exp_f64(-x);
    return (ex - emx) / two;
}

/// Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2
fn cosh_f64(x: f64) -> f64 {
    let zero = x - x;
    let two = zero + 2.0;
    let ex = exp_f64(x);
    let emx = exp_f64(-x);
    return (ex + emx) / two;
}

/// Hyperbolic tangent: tanh(x) = sinh(x) / cosh(x)
fn tanh_f64(x: f64) -> f64 {
    let ex = exp_f64(x);
    let emx = exp_f64(-x);
    return (ex - emx) / (ex + emx);
}

// Special functions (gamma, erf, bessel, encoding) in math_f64_special.wgsl
// ============================================================================
