// SPDX-License-Identifier: AGPL-3.0-or-later
// GPU-resident SU(3) algebra momentum generation via PCG hash PRNG.
//
// Each thread generates one anti-Hermitian traceless 3×3 complex matrix
// (18 f64) from 8 Gaussian random numbers using Box-Muller.
//
// PRNG: PCG hash on (link_idx ⊕ seed ⊕ traj_id) — counter-based,
// no inter-thread state, trivially parallel.
//
// Absorbed from hotSpring lattice QCD (Feb 2026).

struct Params {
    n_links: u32,
    traj_id: u32,
    seed_lo: u32,
    seed_hi: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> momenta: array<f64>;

fn pcg_hash(inp: u32) -> u32 {
    var state = inp * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_u32(link_idx: u32, seq: u32) -> u32 {
    let key = link_idx ^ (params.seed_lo * 2654435761u) ^ (params.traj_id * 2246822519u);
    return pcg_hash(pcg_hash(key + seq) ^ params.seed_hi);
}

fn uniform_f64(link_idx: u32, seq: u32) -> f64 {
    let v = hash_u32(link_idx, seq);
    return (f64(v) + f64(0.5)) * f64(2.3283064365386963e-10);
}

fn box_muller_cos(u1: f64, u2: f64) -> f64 {
    var safe = u1;
    if safe < f64(1e-20) { safe = f64(1e-20); }
    let r = sqrt(f64(-2.0) * log(safe));
    let theta = f64(6.283185307179586) * u2;
    return r * cos(theta);
}

fn gaussian(link_idx: u32, pair: u32) -> f64 {
    let u1 = uniform_f64(link_idx, pair * 2u);
    let u2 = uniform_f64(link_idx, pair * 2u + 1u);
    return box_muller_cos(u1, u2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_links { return; }

    let scale = f64(0.7071067811865476);
    let inv_sqrt3 = f64(0.5773502691896258);
    let a3 = scale * gaussian(idx, 0u);
    let a8 = scale * gaussian(idx, 1u);
    let re_01 = scale * gaussian(idx, 2u);
    let im_01 = scale * gaussian(idx, 3u);
    let re_02 = scale * gaussian(idx, 4u);
    let im_02 = scale * gaussian(idx, 5u);
    let re_12 = scale * gaussian(idx, 6u);
    let im_12 = scale * gaussian(idx, 7u);

    let h00 = a3 + a8 * inv_sqrt3;
    let h11 = f64(-1.0) * a3 + a8 * inv_sqrt3;
    let h22 = f64(-2.0) * a8 * inv_sqrt3;

    let base = idx * 18u;
    let zero = f64(0.0);

    // P = i*H (anti-Hermitian traceless): i*(a+bi) = (-b, a)
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
