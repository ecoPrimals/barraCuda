// SPDX-License-Identifier: AGPL-3.0-or-later
// Link update: U = exp(dt * P) * U via Cayley + Gram-Schmidt reunitarize (DF64 path).
//
// Silicon saturation: runs on FP32 cores via DF64 arithmetic, freeing
// FP64 units for precision-critical reductions (plaquette, KE, deltaH).
// Cayley: exp(A) ≈ (I + A/2)(I - A/2)^{-1}, exact for anti-Hermitian.
// Buffers remain f64 — conversion at load/store boundary only.
// Validated: |deltaP| = 1.44e-7 vs native at 16^4 (RTX 3090).
//
// Prepend: su3_df64_preamble (provides Df64, Cdf64, cdf64_*, su3_mul_df64,
//          su3_identity_df64, sqrt_df64, df64_div, df64_neg, etc.)
//
// Binding layout matches hmc_leapfrog_f64.wgsl for drop-in dispatch.

struct LeapfrogParams {
    volume:  u32,
    n_links: u32,
    _pad0:   u32,
    _pad1:   u32,
    dt:      f64,
    _padf:   f64,
}

@group(0) @binding(0) var<uniform>             params:    LeapfrogParams;
@group(0) @binding(1) var<storage, read_write> links:     array<f64>;
@group(0) @binding(2) var<storage, read_write> momenta:   array<f64>;
@group(0) @binding(3) var<storage, read>       force:     array<f64>;
@group(0) @binding(4) var<storage, read_write> rng_state: array<u32>;

fn cdf64_scale_link(a: Cdf64, s: Df64) -> Cdf64 {
    return Cdf64(df64_mul(a.re, s), df64_mul(a.im, s));
}

fn cdf64_neg_link(a: Cdf64) -> Cdf64 {
    return Cdf64(df64_neg(a.re), df64_neg(a.im));
}

fn su3_inv_df64(a: array<Cdf64, 9>) -> array<Cdf64, 9> {
    var av = a;
    let c00 = cdf64_sub(cdf64_mul(av[4], av[8]), cdf64_mul(av[5], av[7]));
    let c01 = cdf64_sub(cdf64_mul(av[5], av[6]), cdf64_mul(av[3], av[8]));
    let c02 = cdf64_sub(cdf64_mul(av[3], av[7]), cdf64_mul(av[4], av[6]));
    let c10 = cdf64_sub(cdf64_mul(av[2], av[7]), cdf64_mul(av[1], av[8]));
    let c11 = cdf64_sub(cdf64_mul(av[0], av[8]), cdf64_mul(av[2], av[6]));
    let c12 = cdf64_sub(cdf64_mul(av[1], av[6]), cdf64_mul(av[0], av[7]));
    let c20 = cdf64_sub(cdf64_mul(av[1], av[5]), cdf64_mul(av[2], av[4]));
    let c21 = cdf64_sub(cdf64_mul(av[2], av[3]), cdf64_mul(av[0], av[5]));
    let c22 = cdf64_sub(cdf64_mul(av[0], av[4]), cdf64_mul(av[1], av[3]));

    let det = cdf64_add(cdf64_add(cdf64_mul(av[0], c00), cdf64_mul(av[1], c01)), cdf64_mul(av[2], c02));

    let det_conj = cdf64_conj(det);
    let det_norm2 = df64_add(df64_mul(det.re, det.re), df64_mul(det.im, det.im));
    let inv_norm2 = df64_div(df64_from_f32(1.0), det_norm2);
    let inv_det = Cdf64(df64_mul(det_conj.re, inv_norm2), df64_mul(det_conj.im, inv_norm2));

    var r: array<Cdf64, 9>;
    r[0] = cdf64_mul(c00, inv_det); r[1] = cdf64_mul(c10, inv_det); r[2] = cdf64_mul(c20, inv_det);
    r[3] = cdf64_mul(c01, inv_det); r[4] = cdf64_mul(c11, inv_det); r[5] = cdf64_mul(c21, inv_det);
    r[6] = cdf64_mul(c02, inv_det); r[7] = cdf64_mul(c12, inv_det); r[8] = cdf64_mul(c22, inv_det);
    return r;
}

fn su3_reunitarize_gs_df64(m: array<Cdf64, 9>) -> array<Cdf64, 9> {
    var mv = m;
    var r: array<Cdf64, 9>;

    // Normalize row 0
    var n0 = df64_zero();
    for (var j = 0u; j < 3u; j++) {
        n0 = df64_add(n0, df64_add(df64_mul(mv[j].re, mv[j].re), df64_mul(mv[j].im, mv[j].im)));
    }
    let inv0 = df64_div(df64_from_f32(1.0), sqrt_df64(n0));
    for (var j = 0u; j < 3u; j++) { r[j] = cdf64_scale_link(mv[j], inv0); }

    // Orthogonalize row 1
    var dot = cdf64_zero();
    for (var j = 0u; j < 3u; j++) {
        dot = cdf64_add(dot, cdf64_mul(cdf64_conj(r[j]), mv[3u + j]));
    }
    var row1: array<Cdf64, 3>;
    for (var j = 0u; j < 3u; j++) {
        row1[j] = cdf64_sub(mv[3u + j], cdf64_mul(dot, r[j]));
    }
    var n1 = df64_zero();
    for (var j = 0u; j < 3u; j++) {
        n1 = df64_add(n1, df64_add(df64_mul(row1[j].re, row1[j].re), df64_mul(row1[j].im, row1[j].im)));
    }
    let inv1 = df64_div(df64_from_f32(1.0), sqrt_df64(n1));
    for (var j = 0u; j < 3u; j++) { r[3u + j] = cdf64_scale_link(row1[j], inv1); }

    // Row 2 = conj(row0 × row1)
    r[6] = cdf64_conj(cdf64_sub(cdf64_mul(r[1], r[5]), cdf64_mul(r[2], r[4])));
    r[7] = cdf64_conj(cdf64_sub(cdf64_mul(r[2], r[3]), cdf64_mul(r[0], r[5])));
    r[8] = cdf64_conj(cdf64_sub(cdf64_mul(r[0], r[4]), cdf64_mul(r[1], r[3])));

    return r;
}

@compute @workgroup_size(128)
fn link_update_df64(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_links { return; }

    let base = idx * 18u;

    // Load momentum and link from f64 buffers into DF64
    var p: array<Cdf64, 9>;
    var u: array<Cdf64, 9>;
    for (var i = 0u; i < 9u; i++) {
        p[i] = cdf64_from_f64(momenta[base + i * 2u], momenta[base + i * 2u + 1u]);
        u[i] = cdf64_from_f64(links[base + i * 2u], links[base + i * 2u + 1u]);
    }

    let half_dt = df64_mul(df64_from_f64(params.dt), df64_from_f32(0.5));

    // Cayley: exp(dt*P) ≈ (I + dt/2 * P)(I - dt/2 * P)^{-1}
    var plus = su3_identity_df64();
    var minus = su3_identity_df64();
    for (var i = 0u; i < 9u; i++) {
        let h = cdf64_scale_link(p[i], half_dt);
        plus[i] = cdf64_add(plus[i], h);
        minus[i] = cdf64_sub(minus[i], h);
    }

    let inv_m = su3_inv_df64(minus);
    let exp_p = su3_mul_df64(plus, inv_m);
    let new_u = su3_reunitarize_gs_df64(su3_mul_df64(exp_p, u));

    // Store back as f64
    for (var i = 0u; i < 9u; i++) {
        links[base + i * 2u] = df64_to_f64(new_u[i].re);
        links[base + i * 2u + 1u] = df64_to_f64(new_u[i].im);
    }
}
