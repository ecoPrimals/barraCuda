// SPDX-License-Identifier: AGPL-3.0-or-later
// solvation_energy_df64.wgsl — Implicit solvation (GBSA-style, per-atom contribution)
//
// GBSA: DeltaG_solv = -0.5 * (1/eps_in - 1/eps_out) * sum_i q_i^2 / R_i
// R_i = effective Born radius. Per-atom contribution: -0.5 * (1/eps_in - 1/eps_out) * q^2/R
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
// DF64_POLYFILL_PLACEHOLDER

struct SolvationParams {
    n_atoms: u32,
    prefactor: f32,  // -0.5 * (1/eps_in - 1/eps_out)
}

@group(0) @binding(0) var<storage, read> charges: array<vec2<f32>>;  // [N] per-atom charge
@group(0) @binding(1) var<storage, read> born_radii: array<vec2<f32>>;  // [N] effective R
@group(0) @binding(2) var<storage, read_write> energies: array<vec2<f32>>;  // [N] per-atom solvation
@group(0) @binding(3) var<uniform> params: SolvationParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_atoms { return; }

    let q = Df64(charges[idx].x, charges[idx].y);
    let r = Df64(born_radii[idx].x, born_radii[idx].y);

    let r_min = df64_from_f32(0.01);
    var r_safe = r;
    if df64_lt(r, r_min) {
        r_safe = r_min;
    }

    let pref = df64_from_f32(params.prefactor);
    let q2 = df64_mul(q, q);
    let v = df64_mul(pref, df64_div(q2, r_safe));
    energies[idx] = vec2<f32>(v.hi, v.lo);
}
