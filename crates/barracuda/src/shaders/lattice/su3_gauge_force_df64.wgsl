// SPDX-License-Identifier: AGPL-3.0-only
// SU(3) gauge force (DF64 core streaming) — hotSpring neighbor-buffer indexing.
//
// Prepend: df64_core.wgsl + su3_df64.wgsl (for Df64/Cdf64 types and su3_*_df64 ops)
//
// HYBRID PRECISION STRATEGY:
//   DF64 zone (FP32 cores, ~10x throughput):
//     - All staple SU(3) multiplications (6 staples × 3 matmuls each = 18 per link)
//     - Staple sum accumulation
//   f64 zone (FP64 units, precision-critical):
//     - Final W = U * staple and anti-Hermitian traceless projection
//     - Store to output buffer
//
// On RTX 3090: routes 18/19 matmuls through 10,496 FP32 cores instead of
// 164 FP64 units. Net: ~10x throughput for the staple computation.
//
// Buffer layout: UNCHANGED from su3_gauge_force_f64.wgsl (neighbor-buffer indexing).
//
// Cross-spring evolution:
//   hotSpring Exp 012 (Feb 2026) → df64_core.wgsl → toadStool S58
//   → toadStool su3_df64.wgsl + DF64 HMC pipeline
//   → hotSpring v0.6.10 adapts DF64 staple to local neighbor-buffer layout

struct ForceParams {
    volume: u32,
    pad0: u32,
    beta: f64,
}

@group(0) @binding(0) var<uniform> params: ForceParams;
@group(0) @binding(1) var<storage, read> links: array<f64>;
@group(0) @binding(2) var<storage, read> nbr: array<u32>;
@group(0) @binding(3) var<storage, read_write> force: array<f64>;

// ── DF64 link loading (f64 buffer → DF64 SU(3), enters FP32 zone) ──

fn load_su3_df64(off: u32) -> array<Cdf64, 9> {
    var m: array<Cdf64, 9>;
    for (var i = 0u; i < 9u; i++) {
        let idx = off + i * 2u;
        m[i] = cdf64_from_f64(links[idx], links[idx + 1u]);
    }
    return m;
}

// ── f64 link loading (for final multiply, stays on FP64 units) ──

fn load_su3_f64(off: u32) -> array<f64, 18> {
    var m: array<f64, 18>;
    for (var i = 0u; i < 18u; i++) { m[i] = links[off + i]; }
    return m;
}

fn mul_su3_f64(a_in: array<f64, 18>, b_in: array<f64, 18>) -> array<f64, 18> {
    var a = a_in;
    var b = b_in;
    var r: array<f64, 18>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            var re: f64 = f64(0.0);
            var im: f64 = f64(0.0);
            for (var k = 0u; k < 3u; k++) {
                let idx_ik = (i*3u+k)*2u;
                let idx_kj = (k*3u+j)*2u;
                re += a[idx_ik]*b[idx_kj] - a[idx_ik+1u]*b[idx_kj+1u];
                im += a[idx_ik]*b[idx_kj+1u] + a[idx_ik+1u]*b[idx_kj];
            }
            r[(i*3u+j)*2u] = re;
            r[(i*3u+j)*2u+1u] = im;
        }
    }
    return r;
}

fn adj_su3_f64(a_in: array<f64, 18>) -> array<f64, 18> {
    var a = a_in;
    var r: array<f64, 18>;
    for (var i = 0u; i < 3u; i++) {
        for (var j = 0u; j < 3u; j++) {
            r[(i*3u+j)*2u] = a[(j*3u+i)*2u];
            r[(i*3u+j)*2u+1u] = -a[(j*3u+i)*2u+1u];
        }
    }
    return r;
}

// ── DF64 → f64 boundary conversion ──

fn df64_staple_to_f64(m: array<Cdf64, 9>) -> array<f64, 18> {
    var mv = m;
    var r: array<f64, 18>;
    for (var i = 0u; i < 9u; i++) {
        r[i * 2u] = df64_to_f64(mv[i].re);
        r[i * 2u + 1u] = df64_to_f64(mv[i].im);
    }
    return r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let link_idx = idx;
    let n_links = params.volume * 4u;
    if link_idx >= n_links { return; }

    let site = link_idx / 4u;
    let mu = link_idx % 4u;

    // ── DF64 zone: staple sum on FP32 cores ──
    var staple = su3_zero_df64();

    for (var nu = 0u; nu < 4u; nu++) {
        if nu == mu { continue; }

        let fwd_mu = nbr[site * 8u + mu * 2u];
        let fwd_nu = nbr[site * 8u + nu * 2u];
        let bwd_nu = nbr[site * 8u + nu * 2u + 1u];

        // Upper: U_nu(x+mu) * U_mu†(x+nu) * U_nu†(x)
        let u_nu_xmu = load_su3_df64((fwd_mu * 4u + nu) * 18u);
        let u_mu_xnu = load_su3_df64((fwd_nu * 4u + mu) * 18u);
        let u_nu_x   = load_su3_df64((site * 4u + nu) * 18u);
        let upper = su3_mul_df64(su3_mul_df64(u_nu_xmu, su3_adjoint_df64(u_mu_xnu)), su3_adjoint_df64(u_nu_x));

        // Lower: U_nu†(x+mu-nu) * U_mu†(x-nu) * U_nu(x-nu)
        let fwd_mu_bwd_nu = nbr[fwd_mu * 8u + nu * 2u + 1u];
        let u_nu_xmubnu   = load_su3_df64((fwd_mu_bwd_nu * 4u + nu) * 18u);
        let u_mu_xbnu     = load_su3_df64((bwd_nu * 4u + mu) * 18u);
        let u_nu_xbnu     = load_su3_df64((bwd_nu * 4u + nu) * 18u);
        let lower = su3_mul_df64(su3_mul_df64(su3_adjoint_df64(u_nu_xmubnu), su3_adjoint_df64(u_mu_xbnu)), u_nu_xbnu);

        staple = su3_add_df64(staple, su3_add_df64(upper, lower));
    }

    // ── Boundary: DF64 → f64 ──
    let staple_f64 = df64_staple_to_f64(staple);

    // ── f64 zone: final multiply + projection on FP64 units ──
    let u_mu_x = load_su3_f64(link_idx * 18u);
    var w = mul_su3_f64(u_mu_x, staple_f64);

    // Anti-Hermitian traceless projection
    var wdag = adj_su3_f64(w);
    var diff: array<f64, 18>;
    for (var i = 0u; i < 18u; i++) { diff[i] = (w[i] - wdag[i]) * f64(0.5); }

    var tr_re: f64 = diff[0] + diff[8] + diff[16];
    var tr_im: f64 = diff[1] + diff[9] + diff[17];
    let tr3_re = tr_re / f64(3.0);
    let tr3_im = tr_im / f64(3.0);

    var proj: array<f64, 18>;
    for (var i = 0u; i < 18u; i++) { proj[i] = diff[i]; }
    proj[0]  -= tr3_re; proj[1]  -= tr3_im;
    proj[8]  -= tr3_re; proj[9]  -= tr3_im;
    proj[16] -= tr3_re; proj[17] -= tr3_im;

    let s = -params.beta / f64(3.0);
    for (var i = 0u; i < 18u; i++) {
        force[link_idx * 18u + i] = proj[i] * s;
    }
}
