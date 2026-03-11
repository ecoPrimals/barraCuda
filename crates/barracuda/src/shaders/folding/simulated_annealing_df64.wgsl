// SPDX-License-Identifier: AGPL-3.0-only
// simulated_annealing_df64.wgsl — SA acceptance with Metropolis criterion
//
// accept = exp(-(E_new - E_old) / kT) > random(0,1)
// Per-state: input E_old, E_new, kT, random; output accept (1 or 0).
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
// DF64_POLYFILL_PLACEHOLDER

struct SaParams {
    n_states: u32,
    kT: f32,
}

@group(0) @binding(0) var<storage, read> e_old: array<vec2<f32>>;  // [N] old energy as DF64
@group(0) @binding(1) var<storage, read> e_new: array<vec2<f32>>;  // [N] new energy as DF64
@group(0) @binding(2) var<storage, read> random_val: array<f32>;  // [N] uniform(0,1)
@group(0) @binding(3) var<storage, read_write> accept: array<f32>;  // [N] 1 or 0
@group(0) @binding(4) var<uniform> params: SaParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_states { return; }

    let eo = Df64(e_old[idx].x, e_old[idx].y);
    let en = Df64(e_new[idx].x, e_new[idx].y);
    let kT = df64_from_f32(params.kT);

    let delta = df64_sub(en, eo);
    // Always accept if delta <= 0
    if !df64_gt(delta, df64_zero()) {
        accept[idx] = 1.0;
        return;
    }

    let exp_arg = df64_div(delta, kT);
    let prob = exp_df64(df64_neg(exp_arg));
    let r = random_val[idx];
    let r_df = df64_from_f32(r);
    accept[idx] = select(0.0, 1.0, df64_gt(prob, r_df));
}
