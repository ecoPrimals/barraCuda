// Real spherical harmonics Y_l^m(theta, phi) for multipole expansion
// Input: theta and phi interleaved [theta0, phi0, theta1, phi1, ...]
// Params: size, l (degree 0..6), m (order, can be negative)

@group(0) @binding(0) var<storage, read> theta_phi: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    l: u32,
    abs_m: u32,       // |m|
    m_is_positive: u32, // 1 if m>0 (use cos), 0 if m<0 (use sin), ignored when m=0
}

const PI: f32 = 3.14159265359;

// Factorial for n=0..12 (for l+|m| <= 6)
fn factorial(n: u32) -> f32 {
    switch n {
        case 0u: { return 1.0; }
        case 1u: { return 1.0; }
        case 2u: { return 2.0; }
        case 3u: { return 6.0; }
        case 4u: { return 24.0; }
        case 5u: { return 120.0; }
        case 6u: { return 720.0; }
        case 7u: { return 5040.0; }
        case 8u: { return 40320.0; }
        case 9u: { return 362880.0; }
        case 10u: { return 3628800.0; }
        case 11u: { return 39916800.0; }
        case 12u: { return 479001600.0; }
        default: { return 1.0; }
    }
}

// Double factorial (2m-1)!! = 1*3*5*...*(2m-1)
fn double_factorial(m: u32) -> f32 {
    if (m == 0u) {
        return 1.0;
    }
    var r = 1.0;
    for (var k = 1u; k <= m; k = k + 1u) {
        r = r * f32(2u * k - 1u);
    }
    return r;
}

// Associated Legendre P_l^m(x) where x = cos(theta), iterative
fn assoc_legendre(l: u32, m: u32, x: f32) -> f32 {
    let abs_m = m;
    if (abs_m > l) {
        return 0.0;
    }
    let t = 1.0 - x * x;
    if (t <= 0.0) {
        if (abs_m == 0u) {
            return 1.0;
        }
        return 0.0;
    }

    // P_m^m
    var pm = double_factorial(abs_m) * pow(sqrt(t), f32(abs_m));
    if (abs_m % 2u == 1u) {
        pm = -pm;
    }

    if (l == abs_m) {
        return pm;
    }

    // P_{m+1}^m
    var pmp1 = f32(2u * abs_m + 1u) * x * pm;
    if (l == abs_m + 1u) {
        return pmp1;
    }

    // Recurrence for l >= m+2
    var pl_minus2 = pm;
    var pl_minus1 = pmp1;
    for (var ll = abs_m + 2u; ll <= l; ll = ll + 1u) {
        let coef1 = f32(2u * ll - 1u) * x * pl_minus1;
        let coef2 = f32(ll + abs_m - 1u) * pl_minus2;
        let pl = (coef1 - coef2) / f32(ll - abs_m);
        pl_minus2 = pl_minus1;
        pl_minus1 = pl;
    }
    return pl_minus1;
}

// Normalization N_lm = sqrt((2l+1)/(4*pi) * (l-|m|)!/(l+|m|)!)
fn norm_lm(l: u32, abs_m: u32) -> f32 {
    let num = factorial(l - abs_m);
    let den = factorial(l + abs_m);
    let ratio = num / den;
    let pre = f32(2u * l + 1u) / (4.0 * PI);
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
        output[idx] = 0.0;
        return;
    }

    let x = cos(theta);
    let plm = assoc_legendre(l, abs_m, x);

    var angular: f32;
    if (abs_m == 0u) {
        angular = 1.0;
    } else if (m_is_positive == 1u) {
        angular = cos(f32(abs_m) * phi);
    } else {
        angular = sin(f32(abs_m) * phi);
    }

    let n_lm = norm_lm(l, abs_m);
    var y_lm = n_lm * plm * angular;

    if (abs_m != 0u) {
        y_lm = y_lm * sqrt(2.0); // Real form: sqrt(2) for m != 0
    }

    output[idx] = y_lm;
}
