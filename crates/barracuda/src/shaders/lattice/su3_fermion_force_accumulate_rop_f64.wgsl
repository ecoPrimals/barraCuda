// SPDX-License-Identifier: AGPL-3.0-only
//
// ROP-accelerated fermion force accumulation (Tier 3 silicon routing).
//
// Absorbed from hotSpring V0632 (March 2026).
//
// Same physics as staggered_fermion_force_f64.wgsl, but instead of writing
// to a separate force buffer, atomically accumulates weighted force (alpha_s*dt*F)
// into a fixed-point i32 buffer via hardware atomicAdd through the ROP units.
//
// Multiple poles can dispatch simultaneously — no barriers between poles.
// After all poles complete, a conversion kernel adds the accumulated i32
// fixed-point values back to the f64 momentum buffer.
//
// Fixed-point: scale = 2^20 = 1048576. Range +/-2047, precision ~10^-6.
// Sufficient for force accumulation where integrator error is O(dt^2) ~ O(10^-4).

struct Params {
    volume: u32,
    pad0: u32,
    alpha_dt_hi: u32,
    alpha_dt_lo: u32,
    scale_factor: f64,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> links: array<f64>;
@group(0) @binding(2) var<storage, read> x_field: array<f64>;
@group(0) @binding(3) var<storage, read> y_field: array<f64>;
@group(0) @binding(4) var<storage, read> nbr: array<u32>;
@group(0) @binding(5) var<storage, read> phases: array<f64>;
@group(0) @binding(6) var<storage, read_write> force_accum: array<atomic<i32>>;

fn unpack_f64(hi: u32, lo: u32) -> f64 {
    return bitcast<f64>(vec2<u32>(lo, hi));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    let site = idx;
    if site >= params.volume { return; }

    let alpha_dt = unpack_f64(params.alpha_dt_hi, params.alpha_dt_lo);
    let xp = site * 6u;

    for (var mu: u32 = 0u; mu < 4u; mu = mu + 1u) {
        let link_idx = site * 4u + mu;
        let eta = phases[site * 4u + mu];

        let fwd = nbr[site * 8u + mu * 2u];
        let fp = fwd * 6u;

        let neg_eta = f64(0.0) - eta;
        var m: array<f64, 18>;
        for (var a = 0u; a < 3u; a++) {
            let xa_fwd_re = x_field[fp + a*2u];
            let xa_fwd_im = x_field[fp + a*2u + 1u];
            let ya_fwd_re = y_field[fp + a*2u];
            let ya_fwd_im = y_field[fp + a*2u + 1u];

            for (var b = 0u; b < 3u; b++) {
                let yb_here_re = y_field[xp + b*2u];
                let yb_here_im = y_field[xp + b*2u + 1u];
                let xb_here_re = x_field[xp + b*2u];
                let xb_here_im = x_field[xp + b*2u + 1u];

                let t1_re = xa_fwd_re * yb_here_re + xa_fwd_im * yb_here_im;
                let t1_im = xa_fwd_im * yb_here_re - xa_fwd_re * yb_here_im;
                let t2_re = ya_fwd_re * xb_here_re + ya_fwd_im * xb_here_im;
                let t2_im = ya_fwd_im * xb_here_re - ya_fwd_re * xb_here_im;

                let moff = (a * 3u + b) * 2u;
                m[moff]     = neg_eta * (t1_re - t2_re);
                m[moff + 1u] = neg_eta * (t1_im - t2_im);
            }
        }

        let ul = link_idx * 18u;
        var w: array<f64, 18>;
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < 3u; j++) {
                var wr: f64 = f64(0.0);
                var wi: f64 = f64(0.0);
                for (var k = 0u; k < 3u; k++) {
                    let u_re = links[ul + (i*3u+k)*2u];
                    let u_im = links[ul + (i*3u+k)*2u + 1u];
                    let m_re = m[(k*3u+j)*2u];
                    let m_im = m[(k*3u+j)*2u + 1u];
                    wr += u_re * m_re - u_im * m_im;
                    wi += u_re * m_im + u_im * m_re;
                }
                w[(i*3u+j)*2u]     = wr;
                w[(i*3u+j)*2u + 1u] = wi;
            }
        }

        var ta: array<f64, 18>;
        for (var i = 0u; i < 3u; i++) {
            for (var j = 0u; j < 3u; j++) {
                let w_ij_re = w[(i*3u+j)*2u];
                let w_ij_im = w[(i*3u+j)*2u + 1u];
                let w_ji_re = w[(j*3u+i)*2u];
                let w_ji_im = w[(j*3u+i)*2u + 1u];
                let ah_re = f64(0.5) * (w_ij_re - w_ji_re);
                let ah_im = f64(0.5) * (w_ij_im + w_ji_im);
                ta[(i*3u+j)*2u]     = ah_re;
                ta[(i*3u+j)*2u + 1u] = ah_im;
            }
        }

        let tr_re = (ta[0u] + ta[8u] + ta[16u]) / f64(3.0);
        let tr_im = (ta[1u] + ta[9u] + ta[17u]) / f64(3.0);
        ta[0u]  -= tr_re; ta[1u]  -= tr_im;
        ta[8u]  -= tr_re; ta[9u]  -= tr_im;
        ta[16u] -= tr_re; ta[17u] -= tr_im;

        let out_base = link_idx * 18u;
        for (var i = 0u; i < 18u; i++) {
            let val = alpha_dt * ta[i];
            let fixed = i32(val * params.scale_factor);
            atomicAdd(&force_accum[out_base + i], fixed);
        }
    }
}
