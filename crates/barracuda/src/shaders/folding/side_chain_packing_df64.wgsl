// SPDX-License-Identifier: AGPL-3.0-or-later
// side_chain_packing_df64.wgsl — Rotamer scoring for side chain placement
//
// E_rotamer = sum over atom pairs: LJ + Coulomb + clash penalty
// Simplified: score = -log(1 + exp(-E)) for softmax-like ranking.
// Per-rotamer energy input, output score for selection.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct PackingParams {
    n_rotamers: u32,
}

@group(0) @binding(0) var<storage, read> energies: array<vec2<f32>>;  // [N] rotamer energies as DF64
@group(0) @binding(1) var<storage, read_write> scores: array<vec2<f32>>;  // [N] selection scores
@group(0) @binding(2) var<uniform> params: PackingParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_rotamers { return; }

    let e = Df64(energies[idx].x, energies[idx].y);
    // Softplus-style score: -log(1 + exp(-E)) = log_sigmoid(E), higher E -> higher score
    let neg_e = df64_neg(e);
    let exp_neg = exp_df64(neg_e);
    let one = df64_from_f32(1.0);
    let log_arg = df64_add(one, exp_neg);
    let score = df64_neg(log_df64(log_arg));
    scores[idx] = vec2<f32>(score.hi, score.lo);
}
