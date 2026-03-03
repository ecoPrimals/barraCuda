// su3_extended_f64.wgsl — Extended SU(3) operations for HMC / lattice init
//
// Prepend: complex_f64.wgsl + su3.wgsl + lcg_f64.wgsl
//
// Provides: reunitarize, exp_cayley, norm_sq, random_near_identity,
//           random_algebra, su3_sub, su3_store, su3_load_from
//
// These extend the base su3.wgsl functions for dynamical lattice QCD.

const DIVISION_GUARD: f64 = f64(1e-30);
const SQRT3_INV: f64 = f64(0.5773502691896258); // 1/sqrt(3)
const FRAC_1_SQRT2: f64 = f64(0.7071067811865476);

fn su3_sub(a: array<vec2<f64>, 9>, b: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    var av = a;
    var bv = b;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = c64_sub(av[i], bv[i]);
    }
    return r;
}

fn su3_norm_sq(m: array<vec2<f64>, 9>) -> f64 {
    var mv = m;
    var s: f64 = f64(0);
    for (var i = 0u; i < 9u; i = i + 1u) {
        s += c64_abs_sq(mv[i]);
    }
    return s;
}

fn su3_scale_complex(m: array<vec2<f64>, 9>, s: vec2<f64>) -> array<vec2<f64>, 9> {
    var mv = m;
    var r: array<vec2<f64>, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = c64_mul(mv[i], s);
    }
    return r;
}

// Gram-Schmidt reunitarization: normalize row 0, orthogonalize row 1, cross-product row 2.
fn su3_reunitarize(m: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    var u = m;

    // Normalize row 0
    var n0_sq = c64_abs_sq(u[0]) + c64_abs_sq(u[1]) + c64_abs_sq(u[2]);
    var n0 = sqrt(n0_sq);
    if (n0 > DIVISION_GUARD) {
        let inv = f64(1.0) / n0;
        u[0] = c64_scale(u[0], inv);
        u[1] = c64_scale(u[1], inv);
        u[2] = c64_scale(u[2], inv);
    }

    // Orthogonalize row 1 against row 0: row1 -= <row0|row1> * row0
    let dot01 = c64_add(c64_add(
        c64_mul(c64_conj(u[0]), u[3]),
        c64_mul(c64_conj(u[1]), u[4])),
        c64_mul(c64_conj(u[2]), u[5])
    );
    u[3] = c64_sub(u[3], c64_mul(u[0], dot01));
    u[4] = c64_sub(u[4], c64_mul(u[1], dot01));
    u[5] = c64_sub(u[5], c64_mul(u[2], dot01));

    // Normalize row 1
    var n1_sq = c64_abs_sq(u[3]) + c64_abs_sq(u[4]) + c64_abs_sq(u[5]);
    var n1 = sqrt(n1_sq);
    if (n1 > DIVISION_GUARD) {
        let inv = f64(1.0) / n1;
        u[3] = c64_scale(u[3], inv);
        u[4] = c64_scale(u[4], inv);
        u[5] = c64_scale(u[5], inv);
    }

    // Row 2 = conj(row0 × row1)
    u[6] = c64_conj(c64_sub(c64_mul(u[1], u[5]), c64_mul(u[2], u[4])));
    u[7] = c64_conj(c64_sub(c64_mul(u[2], u[3]), c64_mul(u[0], u[5])));
    u[8] = c64_conj(c64_sub(c64_mul(u[0], u[4]), c64_mul(u[1], u[3])));

    return u;
}

// Second-order Cayley approximation: exp(dt*P) ≈ I + dt*P + (dt*P)²/2
fn su3_exp_cayley(p: array<vec2<f64>, 9>, dt: f64) -> array<vec2<f64>, 9> {
    var dp = su3_scale(p, dt);
    var dp2 = su3_mul(dp, dp);
    var r = su3_identity();
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = c64_add(c64_add(r[i], dp[i]), c64_scale(dp2[i], f64(0.5)));
    }
    return r;
}

// Generate SU(3) matrix near identity via Gell-Mann basis expansion.
fn su3_random_near_identity(state: ptr<function, u32>, epsilon: f64) -> array<vec2<f64>, 9> {
    // Diagonal Gell-Mann components
    let a3 = prng_gaussian(state) * epsilon;
    let a8 = prng_gaussian(state) * epsilon;

    // Build Hermitian generator h
    var h: array<vec2<f64>, 9>;
    let zero = f64(0);
    h[0] = c64_new(a3 + a8 * SQRT3_INV, zero);
    h[4] = c64_new(-a3 + a8 * SQRT3_INV, zero);
    h[8] = c64_new(f64(-2.0) * a8 * SQRT3_INV, zero);

    // Off-diagonal: (0,1), (0,2), (1,2) — Hermitian pairs
    for (var pair = 0u; pair < 3u; pair = pair + 1u) {
        let re = prng_gaussian(state) * epsilon;
        let im = prng_gaussian(state) * epsilon;
        switch (pair) {
            case 0u: {
                h[1] = c64_new(re, im);
                h[3] = c64_new(re, -im);
            }
            case 1u: {
                h[2] = c64_new(re, im);
                h[6] = c64_new(re, -im);
            }
            default: {
                h[5] = c64_new(re, im);
                h[7] = c64_new(re, -im);
            }
        }
    }

    // result = I + i*h - h²/2
    var ih = su3_scale_complex(h, c64_new(zero, f64(1.0)));
    var h2 = su3_mul(h, h);
    var r = su3_identity();
    for (var i = 0u; i < 9u; i = i + 1u) {
        r[i] = c64_sub(c64_add(r[i], ih[i]), c64_scale(h2[i], f64(0.5)));
    }
    return su3_reunitarize(r);
}

// Generate random traceless anti-Hermitian su(3) algebra element.
fn su3_random_algebra(state: ptr<function, u32>) -> array<vec2<f64>, 9> {
    let a3 = prng_gaussian(state) * FRAC_1_SQRT2;
    let a8 = prng_gaussian(state) * FRAC_1_SQRT2;

    let zero2 = f64(0);
    var h: array<vec2<f64>, 9>;
    h[0] = c64_new(a3 + a8 * SQRT3_INV, zero2);
    h[4] = c64_new(-a3 + a8 * SQRT3_INV, zero2);
    h[8] = c64_new(f64(-2.0) * a8 * SQRT3_INV, zero2);

    for (var pair = 0u; pair < 3u; pair = pair + 1u) {
        let re = prng_gaussian(state) * FRAC_1_SQRT2;
        let im = prng_gaussian(state) * FRAC_1_SQRT2;
        switch (pair) {
            case 0u: { h[1] = c64_new(re, im); h[3] = c64_new(re, -im); }
            case 1u: { h[2] = c64_new(re, im); h[6] = c64_new(re, -im); }
            default: { h[5] = c64_new(re, im); h[7] = c64_new(re, -im); }
        }
    }

    // Multiply by i to get anti-Hermitian
    return su3_scale_complex(h, c64_new(zero2, f64(1.0)));
}

// NOTE: su3_store/su3_load_from are inlined in each shader because WGSL
// does not allow passing storage buffer pointers as function arguments.
