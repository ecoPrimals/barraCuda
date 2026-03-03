// Real spherical harmonics Y_l^m(theta, phi) — f64 precision
// Input: theta and phi interleaved [theta0, phi0, theta1, phi1, ...]
// Params: size, l (degree 0..6), m (order, can be negative)
//
// **f64 precision**: Required for high-order multipole expansion accuracy

@group(0) @binding(0) var<storage, read> theta_phi: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    l: u32,
    abs_m: u32,       // |m|
    m_is_positive: u32, // 1 if m>0 (use cos), 0 if m<0 (use sin), ignored when m=0
}

const PI: f64 = 3.14159265358979323846;
const SQRT2: f64 = 1.4142135623730950488;  // sqrt(2) — avoids sqrt(f64) Naga overload issue

// Factorial for n=0..20 (extended for f64 precision needs)
fn factorial(n: u32) -> f64 {
    switch n {
        case 0u: { return f64(1.0); }
        case 1u: { return f64(1.0); }
        case 2u: { return f64(2.0); }
        case 3u: { return f64(6.0); }
        case 4u: { return f64(24.0); }
        case 5u: { return f64(120.0); }
        case 6u: { return f64(720.0); }
        case 7u: { return f64(5040.0); }
        case 8u: { return f64(40320.0); }
        case 9u: { return f64(362880.0); }
        case 10u: { return f64(3628800.0); }
        case 11u: { return f64(39916800.0); }
        case 12u: { return f64(479001600.0); }
        case 13u: { return f64(6227020800.0); }
        case 14u: { return f64(87178291200.0); }
        case 15u: { return f64(1307674368000.0); }
        case 16u: { return f64(20922789888000.0); }
        case 17u: { return f64(355687428096000.0); }
        case 18u: { return f64(6402373705728000.0); }
        case 19u: { return f64(121645100408832000.0); }
        case 20u: { return f64(2432902008176640000.0); }
        default: { return f64(1.0); }
    }
}

// Double factorial (2m-1)!! = 1*3*5*...*(2m-1)
fn double_factorial(m: u32) -> f64 {
    if (m == 0u) {
        return f64(1.0);
    }
    var r: f64 = f64(1.0);
    for (var k = 1u; k <= m; k = k + 1u) {
        let term = 2u * k - 1u;
        r = r * f64(term);
    }
    return r;
}

// Associated Legendre P_l^m(x) where x = cos(theta), iterative
fn assoc_legendre(l: u32, m: u32, x: f64) -> f64 {
    let abs_m = m;
    if (abs_m > l) {
        return f64(0.0);
    }
    let t = f64(1.0) - x * x;
    if (t <= f64(0.0)) {
        if (abs_m == 0u) {
            // P_l^0(1) = 1 for all l; P_l^0(-1) = (-1)^l
            if (x < f64(0.0) && l % 2u == 1u) {
                return f64(-1.0);
            }
            return f64(1.0);
        }
        return f64(0.0);
    }

    // P_m^m
    var pm = double_factorial(abs_m) * pow(sqrt(t), f64(abs_m));
    if (abs_m % 2u == 1u) {
        pm = -pm;
    }

    if (l == abs_m) {
        return pm;
    }

    // P_{m+1}^m
    let term_2m1 = 2u * abs_m + 1u;
    var pmp1 = f64(term_2m1) * x * pm;
    if (l == abs_m + 1u) {
        return pmp1;
    }

    // Recurrence for l >= m+2
    var pl_minus2 = pm;
    var pl_minus1 = pmp1;
    for (var ll = abs_m + 2u; ll <= l; ll = ll + 1u) {
        let term1 = 2u * ll - 1u;
        let term2 = ll + abs_m - 1u;
        let coef1 = f64(term1) * x * pl_minus1;
        let coef2 = f64(term2) * pl_minus2;
        let pl = (coef1 - coef2) / f64(ll - abs_m);
        pl_minus2 = pl_minus1;
        pl_minus1 = pl;
    }
    return pl_minus1;
}

// Normalization N_lm = sqrt((2l+1)/(4*pi) * (l-|m|)!/(l+|m|)!)
fn norm_lm(l: u32, abs_m: u32) -> f64 {
    let num = factorial(l - abs_m);
    let den = factorial(l + abs_m);
    let ratio = num / den;
    let term = 2u * l + 1u;
    let pre = f64(term) / (f64(4.0) * PI);
    return sqrt(pre * ratio);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let theta = theta_phi[idx * 2u];
    let phi = theta_phi[idx * 2u + 1u];

    let l = params.l;
    let abs_m = params.abs_m;
    let m_is_positive = params.m_is_positive;

    if (abs_m > l) {
        output[idx] = f64(0.0);
        return;
    }

    let x = cos(theta);
    let plm = assoc_legendre(l, abs_m, x);

    var angular: f64;
    if (abs_m == 0u) {
        angular = f64(1.0);
    } else if (m_is_positive == 1u) {
        angular = cos(f64(abs_m) * phi);
    } else {
        angular = sin(f64(abs_m) * phi);
    }

    let n_lm = norm_lm(l, abs_m);
    var y_lm = n_lm * plm * angular;

    if (abs_m != 0u) {
        y_lm = y_lm * SQRT2;  // Real form: sqrt(2) for m != 0
    }

    output[idx] = y_lm;
}
