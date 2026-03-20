// SPDX-License-Identifier: AGPL-3.0-or-later
// su3_hmc_force_df64.wgsl — Hybrid SU(3) HMC gauge force (DF64 core streaming)
//
// Prepend: complex_f64.wgsl + su3.wgsl + df64_core.wgsl + su3_df64.wgsl
//
// HYBRID PRECISION STRATEGY:
//   DF64 zone (FP32 cores, ~10x throughput):
//     - All staple SU(3) multiplications (6 staples × 3 matmuls each = 18 per link)
//     - Staple sum accumulation
//   f64 zone (FP64 units, precision-critical):
//     - Final F_raw = staple_sum × U†  (1 matmul)
//     - su(3) algebra projection (anti-Hermitian traceless)
//     - Store to output buffer
//
// On RTX 3090: routes 18/19 matmuls through 10,496 FP32 cores instead of
// 164 FP64 units. Net: ~10x throughput for the staple computation (40% of HMC).
//
// Buffer layout: UNCHANGED from su3_hmc_force_f64.wgsl.
//   links[V × 4 × 18]: gauge links (f64)
//   force[V × 4 × 18]: output force (f64)
//
// hotSpring core-streaming discovery (Feb 2026). Production wiring: toadStool.

struct ForceParams {
    nt:     u32,
    nx:     u32,
    ny:     u32,
    nz:     u32,
    volume: u32,
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
    beta:   f64,
    _padf:  f64,
    _padf1: f64,
    _padf2: f64,
}

@group(0) @binding(0) var<uniform>             params: ForceParams;
@group(0) @binding(1) var<storage, read>       links:  array<f64>; // [V×4×18]
@group(0) @binding(2) var<storage, read_write> force:  array<f64>; // [V×4×18]

// ── Coordinate helpers ───────────────────────────────────────────────────────

fn site_to_coords(s: u32) -> vec4<u32> {
    let nxyz = params.nx * params.ny * params.nz;
    let nyz  = params.ny * params.nz;
    let t    = s / nxyz;
    let rem  = s % nxyz;
    let x    = rem / nyz;
    let rem2 = rem % nyz;
    let y    = rem2 / params.nz;
    let z    = rem2 % params.nz;
    return vec4<u32>(t, x, y, z);
}

fn coords_to_site(c: vec4<u32>) -> u32 {
    return c.x * (params.nx * params.ny * params.nz)
         + c.y * (params.ny * params.nz)
         + c.z * params.nz
         + c.w;
}

fn shift_fwd(c: vec4<u32>, mu: u32) -> vec4<u32> {
    var r = c;
    switch (mu) {
        case 0u: { r.x = (c.x + 1u) % params.nt; }
        case 1u: { r.y = (c.y + 1u) % params.nx; }
        case 2u: { r.z = (c.z + 1u) % params.ny; }
        default: { r.w = (c.w + 1u) % params.nz; }
    }
    return r;
}

fn shift_bwd(c: vec4<u32>, mu: u32) -> vec4<u32> {
    var r = c;
    switch (mu) {
        case 0u: { r.x = (c.x + params.nt - 1u) % params.nt; }
        case 1u: { r.y = (c.y + params.nx - 1u) % params.nx; }
        case 2u: { r.z = (c.z + params.ny - 1u) % params.ny; }
        default: { r.w = (c.w + params.nz - 1u) % params.nz; }
    }
    return r;
}

// ── DF64 link loading (f64 buffer → DF64 SU(3)) ─────────────────────────────
// Boundary: f64 → DF64 at load. Runs on FP32 cores from here.

fn load_link_df64(site: u32, mu: u32) -> array<Cdf64, 9> {
    var m: array<Cdf64, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = cdf64_from_f64(links[off], links[off + 1u]);
    }
    return m;
}

// ── f64 link loading (for final multiply — runs on FP64 units) ──────────────

fn load_link_f64(site: u32, mu: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = c64_new(links[off], links[off + 1u]);
    }
    return m;
}

// ── SU(3) algebra projection (f64, precision-critical) ──────────────────────
// Reuses native f64 — this runs on the FP64 units.
// P(M) = (M - M†)/2 − Tr(M - M†)/6 · I

fn su3_project_algebra(m: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    var mv = m;
    var ah: array<vec2<f64>, 9>;
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            let mij = mv[i * 3u + j];
            let mji = mv[j * 3u + i];
            ah[i * 3u + j] = vec2<f64>(
                (mij.x - mji.x) * 0.5,
                (mij.y + mji.y) * 0.5,
            );
        }
    }
    let tr_re = (ah[0].x + ah[4].x + ah[8].x) / 3.0;
    let tr_im = (ah[0].y + ah[4].y + ah[8].y) / 3.0;
    ah[0] = vec2<f64>(ah[0].x - tr_re, ah[0].y - tr_im);
    ah[4] = vec2<f64>(ah[4].x - tr_re, ah[4].y - tr_im);
    ah[8] = vec2<f64>(ah[8].x - tr_re, ah[8].y - tr_im);
    return ah;
}

fn store_force(site: u32, mu: u32, f: array<vec2<f64>, 9>) {
    var fv = f;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        force[base + i * 2u]      = fv[i].x;
        force[base + i * 2u + 1u] = fv[i].y;
    }
}

// ── DF64 staple sum (runs on FP32 cores) ────────────────────────────────────
//
// For link U_mu(x), the staple sum is:
//   Σ = Σ_{nu≠mu} [
//         U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)       (forward)
//       + U_nu†(x+mu-nu) * U_mu†(x-nu) * U_nu(x-nu)  (backward)
//   ]
//
// 6 staples total (3 forward + 3 backward), each = 2–3 DF64 SU(3) matmuls.
// This is the hot inner loop — all 18 matmuls per link run on FP32 cores.

fn compute_staple_sum_df64(site: u32, mu: u32, c: vec4<u32>) -> array<Cdf64, 9> {
    var staple_sum = su3_zero_df64();

    let c_mu = shift_fwd(c, mu);

    for (var nu = 0u; nu < 4u; nu = nu + 1u) {
        if (nu == mu) { continue; }

        let c_nu     = shift_fwd(c, nu);
        let c_bwd_nu = shift_bwd(c, nu);
        let c_mu_bnu = shift_bwd(c_mu, nu);

        let s_nu     = coords_to_site(c_nu);
        let s_mu     = coords_to_site(c_mu);
        let s_bwd_nu = coords_to_site(c_bwd_nu);
        let s_mu_bnu = coords_to_site(c_mu_bnu);

        // Forward staple: U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
        let fwd_st = su3_mul_df64(
            su3_mul_df64(load_link_df64(s_mu, nu), su3_adjoint_df64(load_link_df64(s_nu, mu))),
            su3_adjoint_df64(load_link_df64(site, nu)),
        );

        // Backward staple: U_nu†(x+mu-nu) * U_mu†(x-nu) * U_nu(x-nu)
        let bwd_st = su3_mul_df64(
            su3_mul_df64(su3_adjoint_df64(load_link_df64(s_mu_bnu, nu)), su3_adjoint_df64(load_link_df64(s_bwd_nu, mu))),
            load_link_df64(s_bwd_nu, nu),
        );

        staple_sum = su3_add_df64(staple_sum, su3_add_df64(fwd_st, bwd_st));
    }

    return staple_sum;
}

// ── Kernel ───────────────────────────────────────────────────────────────────
//
// Hybrid precision boundary:
//   DF64 (FP32 cores): staple sum computation (18 SU(3) matmuls per link)
//   f64  (FP64 units): staple×U† multiply, algebra projection, store

@compute @workgroup_size(64)
fn hmc_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.volume) { return; }

    let c = site_to_coords(site);
    let coeff = -params.beta / 3.0;

    for (var mu = 0u; mu < 4u; mu = mu + 1u) {
        // ── DF64 zone: staple sum on FP32 cores ──
        let staples_df64 = compute_staple_sum_df64(site, mu, c);

        // ── Boundary: DF64 → f64 ──
        let staples = su3_df64_to_f64(staples_df64);

        // ── f64 zone: final multiply + projection on FP64 units ──
        let u_dag = su3_adjoint(load_link_f64(site, mu));
        let f_raw = su3_mul(staples, u_dag);
        let f_alg = su3_project_algebra(f_raw);
        store_force(site, mu, su3_scale(f_alg, coeff));
    }
}
