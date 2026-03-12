// SPDX-License-Identifier: AGPL-3.0-only
// su3_hmc_force_f64.wgsl — SU(3) HMC gauge force (Wilson action)
//
// Prepend: complex_f64.wgsl + su3.wgsl
//
// Computes the gauge force for each link U_mu(x) from the Wilson action:
//
//   S_W = -beta/3 * Σ_{x,mu<nu} Re Tr(U_p(x,mu,nu))
//
//   Force on link U_mu(x):
//   F_mu(x) = -beta/3 * Im Tr(Σ_staple * U_mu†(x))
//
// where the "staple sum" for (x, mu) consists of 6 staple products
// (3 forward, 3 backward) in the (d-1)=3 perpendicular directions.
//
// Output: force[V × 4 × 18] f64 — Lie-algebra force matrix at each link.
//   The force is stored as an anti-Hermitian traceless matrix (algebra element):
//   f_mu(x) = (F - F†) / 2 - Tr(F - F†)/6   (projection onto su(3) algebra)
//
// Buffer layout:
//   links[V × 4 × 18]: gauge links, same layout as wilson_plaquette_f64.wgsl
//   force[V × 4 × 18]: output force (algebra elements), zeroed before dispatch
//
// hotSpring design: lattice/hmc.rs (v0.5.16, Feb 2026)

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

// ── Coordinate helpers (identical to plaquette shader) ────────────────────────

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

fn load_link(site: u32, mu: u32) -> array<vec2<f64>, 9> {
    var m: array<vec2<f64>, 9>;
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        let off = base + i * 2u;
        m[i] = c64_new(links[off], links[off + 1u]);
    }
    return m;
}

// ── SU(3) algebra projection ──────────────────────────────────────────────────
// Project matrix M onto the su(3) algebra: anti-Hermitian, traceless.
//   P(M) = (M - M†) / 2  −  Tr(M − M†) / 6 · I

fn su3_project_algebra(m: array<vec2<f64>, 9>) -> array<vec2<f64>, 9> {
    var mv = m; // copy to var — Naga requires var for runtime array indexing
    // Anti-Hermitian part element by element:
    //   ah[i,j] = (M[i,j] - conj(M[j,i])) / 2
    // With M[i,j] = (re, im) and conj(M[j,i]) = (mji.x, -mji.y):
    //   ah[i,j].re = (mij.x - mji.x) * 0.5
    //   ah[i,j].im = (mij.y + mji.y) * 0.5
    // Written as raw vec2 ops to avoid Naga nested-call type inference issues.
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
    // Subtract Tr(ah)/3 from diagonal to make traceless.
    let tr_re = (ah[0].x + ah[4].x + ah[8].x) / 3.0;
    let tr_im = (ah[0].y + ah[4].y + ah[8].y) / 3.0;
    ah[0] = vec2<f64>(ah[0].x - tr_re, ah[0].y - tr_im);
    ah[4] = vec2<f64>(ah[4].x - tr_re, ah[4].y - tr_im);
    ah[8] = vec2<f64>(ah[8].x - tr_re, ah[8].y - tr_im);
    return ah;
}

// ── Store algebra element to output buffer ────────────────────────────────────

fn store_force(site: u32, mu: u32, f: array<vec2<f64>, 9>) {
    var fv = f; // copy to var — Naga requires var for runtime array indexing
    let base = (site * 4u + mu) * 18u;
    for (var i = 0u; i < 9u; i = i + 1u) {
        force[base + i * 2u]      = fv[i].x;
        force[base + i * 2u + 1u] = fv[i].y;
    }
}

// ── Staple sum ────────────────────────────────────────────────────────────────
//
// For link U_mu(x), the staple sum is:
//
//   Σ = Σ_{nu≠mu} [
//         U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)       (forward staple)
//       + U_nu†(x+mu-nu) * U_mu†(x-nu) * U_nu(x-nu)  (backward staple)
//   ]
//
// Force = -beta/3 * Im Tr( Σ * U_mu†(x) )  projected onto su(3)

fn compute_staple_sum(site: u32, mu: u32, c: vec4<u32>) -> array<vec2<f64>, 9> {
    var staple_sum = su3_identity();
    for (var i = 0u; i < 9u; i = i + 1u) { staple_sum[i] = c64_zero(); }

    let c_mu = shift_fwd(c, mu);

    for (var nu = 0u; nu < 4u; nu = nu + 1u) {
        if (nu == mu) { continue; }

        let c_nu     = shift_fwd(c, nu);
        let c_mu_nu  = shift_fwd(c_mu, nu);
        let c_bwd_nu = shift_bwd(c, nu);
        let c_mu_bnu = shift_bwd(c_mu, nu);

        let s_nu     = coords_to_site(c_nu);
        let s_mu     = coords_to_site(c_mu);
        let s_mu_nu  = coords_to_site(c_mu_nu);
        let s_bwd_nu = coords_to_site(c_bwd_nu);
        let s_mu_bnu = coords_to_site(c_mu_bnu);

        // Forward staple: U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
        let fwd_st = su3_mul(
            su3_mul(load_link(s_mu, nu), su3_adjoint(load_link(s_nu, mu))),
            su3_adjoint(load_link(site, nu)),
        );

        // Backward staple: U_nu†(x+mu-nu) * U_mu†(x-nu) * U_nu(x-nu)
        let bwd_st = su3_mul(
            su3_mul(su3_adjoint(load_link(s_mu_bnu, nu)), su3_adjoint(load_link(s_bwd_nu, mu))),
            load_link(s_bwd_nu, nu),
        );

        staple_sum = su3_add(staple_sum, su3_add(fwd_st, bwd_st));
    }

    return staple_sum;
}

// ── Kernel ────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn hmc_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    let site = gid.x;
    if (site >= params.volume) { return; }

    let c = site_to_coords(site);
    let coeff = -params.beta / 3.0;

    for (var mu = 0u; mu < 4u; mu = mu + 1u) {
        let staples = compute_staple_sum(site, mu, c);
        let u_dag   = su3_adjoint(load_link(site, mu));
        // F_raw = Σ_staple * U_mu†(x)
        let f_raw   = su3_mul(staples, u_dag);
        // Project onto algebra and scale by -beta/3
        let f_alg   = su3_project_algebra(f_raw);
        store_force(site, mu, su3_scale(f_alg, coeff));
    }
}
