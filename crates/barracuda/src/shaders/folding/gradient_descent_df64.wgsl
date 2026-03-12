// SPDX-License-Identifier: AGPL-3.0-only
// gradient_descent_df64.wgsl — Steepest descent energy minimization step
//
// x_new = x_old - step * grad
// Per-coordinate update. Input: positions, gradients. Output: updated positions.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct GradientParams {
    n_coords: u32,
    step: f32,
}

@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;  // [N] current coords as DF64
@group(0) @binding(1) var<storage, read> gradients: array<vec2<f32>>;  // [N] grad E as DF64
@group(0) @binding(2) var<storage, read_write> out_positions: array<vec2<f32>>;  // [N] updated
@group(0) @binding(3) var<uniform> params: GradientParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_coords { return; }

    let x = Df64(positions[idx].x, positions[idx].y);
    let g = Df64(gradients[idx].x, gradients[idx].y);
    let step = df64_from_f32(params.step);

    let update = df64_mul(step, g);
    let x_new = df64_sub(x, update);
    out_positions[idx] = vec2<f32>(x_new.hi, x_new.lo);
}
