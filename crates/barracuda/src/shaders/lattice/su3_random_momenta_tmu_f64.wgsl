// SPDX-License-Identifier: AGPL-3.0-only
//
// GPU-resident SU(3) algebra momentum generation — TMU-accelerated Box-Muller.
//
// Absorbed from hotSpring V0632 (March 2026).
//
// Same physics as su3_random_momenta_f64.wgsl, but offloads the expensive
// log/cos/sin transcendentals to TMU texture lookups. ALU only handles
// PCG hash, sqrt, and the SU(3) Gell-Mann construction.
//
// TMU lookup tables (created once, reused across trajectories):
//   log_table: R32Float 4096x1 — precomputed -2*log(x/4095)
//   trig_table: Rg32Float 4096x1 — precomputed (cos(2pi*x/4095), sin(2pi*x/4095))
//
// Requires prng_pcg_f64.wgsl prepended for pcg_hash, uniform_f64.

struct Params {
    n_links: u32,
    traj_id: u32,
    seed_lo: u32,
    seed_hi: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> momenta: array<f64>;
@group(0) @binding(2) var log_table: texture_2d<f32>;
@group(0) @binding(3) var trig_table: texture_2d<f32>;

fn box_muller_tmu(link_idx: u32, pair: u32) -> f64 {
    let u1 = uniform_f64(link_idx, pair * 2u);
    let u2 = uniform_f64(link_idx, pair * 2u + 1u);

    let log_idx = u32(clamp(f32(u1), 1.0 / 4095.0, 1.0) * 4095.0);
    let neg2log = f64(textureLoad(log_table, vec2<u32>(log_idx, 0u), 0).r);
    let r = sqrt_f64(neg2log);

    let trig_idx = u32(clamp(f32(u2), 0.0, 1.0) * 4095.0);
    let cs = textureLoad(trig_table, vec2<u32>(trig_idx, 0u), 0);
    let cos_theta = f64(cs.r);

    return r * cos_theta;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let idx = gid.x + gid.y * nwg.x * 64u;
    if idx >= params.n_links { return; }

    let scale = f64(0.7071067811865476);       // 1/sqrt(2)
    let inv_sqrt3 = f64(0.5773502691896258);   // 1/sqrt(3)
    let a3 = scale * box_muller_tmu(idx, 0u);
    let a8 = scale * box_muller_tmu(idx, 1u);
    let re_01 = scale * box_muller_tmu(idx, 2u);
    let im_01 = scale * box_muller_tmu(idx, 3u);
    let re_02 = scale * box_muller_tmu(idx, 4u);
    let im_02 = scale * box_muller_tmu(idx, 5u);
    let re_12 = scale * box_muller_tmu(idx, 6u);
    let im_12 = scale * box_muller_tmu(idx, 7u);

    let h00 = a3 + a8 * inv_sqrt3;
    let h11 = f64(-1.0) * a3 + a8 * inv_sqrt3;
    let h22 = f64(-2.0) * a8 * inv_sqrt3;

    let base = idx * 18u;
    let zero = f64(0.0);

    momenta[base + 0u] = zero;
    momenta[base + 1u] = h00;
    momenta[base + 2u] = f64(-1.0) * im_01;
    momenta[base + 3u] = re_01;
    momenta[base + 4u] = f64(-1.0) * im_02;
    momenta[base + 5u] = re_02;
    momenta[base + 6u] = im_01;
    momenta[base + 7u] = re_01;
    momenta[base + 8u] = zero;
    momenta[base + 9u] = h11;
    momenta[base + 10u] = f64(-1.0) * im_12;
    momenta[base + 11u] = re_12;
    momenta[base + 12u] = im_02;
    momenta[base + 13u] = re_02;
    momenta[base + 14u] = im_12;
    momenta[base + 15u] = re_12;
    momenta[base + 16u] = zero;
    momenta[base + 17u] = h22;
}
