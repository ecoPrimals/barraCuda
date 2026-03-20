// SPDX-License-Identifier: AGPL-3.0-or-later
// GPU-resident Gaussian random fermion field generation via PCG hash PRNG.
//
// Each thread generates one site's fermion field: 3 colors × (re, im) = 6 f64.
// Used for pseudofermion heat bath: η ~ N(0,1), then φ = D†η.
//
// PRNG: PCG hash on (site_idx ⊕ seed ⊕ traj_id) — counter-based,
// no inter-thread state, trivially parallel.
//
// Absorbed from hotSpring lattice QCD (Feb 2026).

struct Params {
    volume: u32,
    traj_id: u32,
    seed_lo: u32,
    seed_hi: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> fermion: array<f64>;

fn pcg_hash(inp: u32) -> u32 {
    var state = inp * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_u32(site_idx: u32, seq: u32) -> u32 {
    let key = site_idx ^ (params.seed_lo * 2654435761u) ^ (params.traj_id * 2246822519u);
    return pcg_hash(pcg_hash(key + seq) ^ params.seed_hi);
}

fn uniform_f64(site_idx: u32, seq: u32) -> f64 {
    let v = hash_u32(site_idx, seq);
    return (f64(v) + f64(0.5)) * f64(2.3283064365386963e-10);
}

fn box_muller_cos(u1: f64, u2: f64) -> f64 {
    var safe = u1;
    if safe < f64(1e-20) { safe = f64(1e-20); }
    let r = sqrt_f64(f64(-2.0) * log_f64(safe));
    let theta = f64(6.283185307179586) * u2;
    return r * cos_f64(theta);
}

fn gaussian(site_idx: u32, pair: u32) -> f64 {
    let u1 = uniform_f64(site_idx, pair * 2u);
    let u2 = uniform_f64(site_idx, pair * 2u + 1u);
    return box_muller_cos(u1, u2);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.volume { return; }

    let base = idx * 6u;
    // 3 colors × (re, im) = 3 Box-Muller pairs
    fermion[base + 0u] = gaussian(idx, 0u);  // color 0 re
    fermion[base + 1u] = gaussian(idx, 1u);  // color 0 im
    fermion[base + 2u] = gaussian(idx, 2u);  // color 1 re
    fermion[base + 3u] = gaussian(idx, 3u);  // color 1 im
    fermion[base + 4u] = gaussian(idx, 4u);  // color 2 re
    fermion[base + 5u] = gaussian(idx, 5u);  // color 2 im
}
