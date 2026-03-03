// SPDX-License-Identifier: AGPL-3.0-or-later
// rmsd_df64.wgsl — Root Mean Square Deviation between two structures (Kabsch alignment)
//
// RMSD = sqrt(mean((x_i - y_i)^2)) after optimal superposition.
// Per-atom contribution: (x_i - y_i)^2 for atom i. Final RMSD requires reduction.
// This shader computes squared deviations; caller reduces and sqrt.
// Alternatively: single workgroup computes full RMSD for small N.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
// DF64_POLYFILL_PLACEHOLDER

struct RmsdParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> x: array<vec2<f32>>;  // [N*3] structure A
@group(0) @binding(1) var<storage, read> y: array<vec2<f32>>;  // [N*3] structure B (aligned)
@group(0) @binding(2) var<storage, read_write> sq_dev: array<vec2<f32>>;  // [N] per-atom squared deviation
@group(0) @binding(3) var<uniform> params: RmsdParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }

    let b = idx * 3u;
    let dx = df64_sub(Df64(x[b].x, x[b].y), Df64(y[b].x, y[b].y));
    let dy = df64_sub(Df64(x[b + 1u].x, x[b + 1u].y), Df64(y[b + 1u].x, y[b + 1u].y));
    let dz = df64_sub(Df64(x[b + 2u].x, x[b + 2u].y), Df64(y[b + 2u].x, y[b + 2u].y));

    let d_sq = df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));
    sq_dev[idx] = vec2<f32>(d_sq.hi, d_sq.lo);
}
