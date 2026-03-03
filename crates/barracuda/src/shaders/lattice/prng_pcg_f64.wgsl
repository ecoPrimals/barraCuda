// SPDX-License-Identifier: AGPL-3.0-only
// Shared PRNG core: PCG hash → uniform f64 on [0, 1).
//
// Requires a `Params` struct with fields { *, traj_id: u32, seed_lo: u32, seed_hi: u32 }
// bound at @group(0) @binding(0). The first field may vary per consumer.
//
// Consumer shaders include this via Rust string concatenation:
//   let wgsl = format!("{header}\n{prng_core}\n{consumer_body}");
// then define their own Params struct + bindings before this library,
// and their own box_muller_cos/gaussian after it (to control f64 vs f32 cos).

fn pcg_hash(inp: u32) -> u32 {
    var state = inp * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash_u32(idx: u32, seq: u32) -> u32 {
    let key = idx ^ (params.seed_lo * 2654435761u) ^ (params.traj_id * 2246822519u);
    return pcg_hash(pcg_hash(key + seq) ^ params.seed_hi);
}

fn uniform_f64(idx: u32, seq: u32) -> f64 {
    let v = hash_u32(idx, seq);
    return (f64(v) + f64(0.5)) / f64(4294967296.0);
}
