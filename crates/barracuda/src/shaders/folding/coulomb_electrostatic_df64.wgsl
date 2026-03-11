// SPDX-License-Identifier: AGPL-3.0-only
// coulomb_electrostatic_df64.wgsl — Coulomb electrostatic energy with dielectric
//
// V(r) = k_e * q_i * q_j / (epsilon_r * r)
// k_e = 1/(4*pi*epsilon_0) in appropriate units. Per pair.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
// DF64_POLYFILL_PLACEHOLDER

struct CoulombParams {
    n_pairs: u32,
    coulomb_const: f32,  // k_e / epsilon_r
}

@group(0) @binding(0) var<storage, read> distances: array<vec2<f32>>;  // [M] pairwise distances
@group(0) @binding(1) var<storage, read> charges: array<vec2<f32>>;  // [2*M] q_i, q_j per pair
@group(0) @binding(2) var<storage, read_write> energies: array<vec2<f32>>;  // [M] Coulomb energy
@group(0) @binding(3) var<uniform> params: CoulombParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_pairs { return; }

    let r = Df64(distances[idx].x, distances[idx].y);
    let qi = Df64(charges[idx * 2u].x, charges[idx * 2u].y);
    let qj = Df64(charges[idx * 2u + 1u].x, charges[idx * 2u + 1u].y);

    let r_min = df64_from_f32(0.1);
    var r_safe = r;
    if df64_lt(r, r_min) {
        r_safe = r_min;
    }

    let k = df64_from_f32(params.coulomb_const);
    let v = df64_div(df64_mul(k, df64_mul(qi, qj)), r_safe);
    energies[idx] = vec2<f32>(v.hi, v.lo);
}
