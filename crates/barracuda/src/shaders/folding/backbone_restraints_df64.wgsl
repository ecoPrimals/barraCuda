// SPDX-License-Identifier: AGPL-3.0-only
// backbone_restraints_df64.wgsl — Harmonic restraints on backbone atoms
//
// E = 0.5 * k * (x - x_ref)^2 per coordinate
// Per-atom or per-coordinate harmonic potential.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
// DF64_POLYFILL_PLACEHOLDER

struct RestraintParams {
    n_coords: u32,
    k: f32,  // force constant
}

@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;  // [N] current as DF64
@group(0) @binding(1) var<storage, read> ref_positions: array<vec2<f32>>;  // [N] target as DF64
@group(0) @binding(2) var<storage, read_write> energies: array<vec2<f32>>;  // [N] per-coord energy
@group(0) @binding(3) var<uniform> params: RestraintParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_coords { return; }

    let x = Df64(positions[idx].x, positions[idx].y);
    let x_ref = Df64(ref_positions[idx].x, ref_positions[idx].y);
    let k = df64_from_f32(params.k);

    let dx = df64_sub(x, x_ref);
    let dx2 = df64_mul(dx, dx);
    let half = df64_from_f32(0.5);
    let e = df64_mul(df64_mul(half, k), dx2);
    energies[idx] = vec2<f32>(e.hi, e.lo);
}
