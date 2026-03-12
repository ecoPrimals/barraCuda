// SPDX-License-Identifier: AGPL-3.0-only
// lennard_jones_df64.wgsl — 12-6 Lennard-Jones potential for van der Waals
//
// V(r) = 4*eps*((sigma/r)^12 - (sigma/r)^6)
// Per pair contribution. Input: pairwise distances [M], sigma, epsilon.
// Output: LJ energy per pair [M].
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct LjParams {
    n_pairs: u32,
    sigma: f32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> distances: array<vec2<f32>>;  // [M] pairwise distances as DF64
@group(0) @binding(1) var<storage, read_write> energies: array<vec2<f32>>;  // [M] LJ energy as DF64
@group(0) @binding(2) var<uniform> params: LjParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_pairs { return; }

    let r = Df64(distances[idx].x, distances[idx].y);
    let sigma = df64_from_f32(params.sigma);
    let eps = df64_from_f32(params.epsilon);

    // Avoid singularity: clamp r to minimum
    let r_min = df64_from_f32(0.1);
    var r_safe = r;
    if df64_lt(r, r_min) {
        r_safe = r_min;
    }

    let sigma_r = df64_div(sigma, r_safe);
    let sigma_r2 = df64_mul(sigma_r, sigma_r);
    let sigma_r6 = df64_mul(df64_mul(sigma_r2, sigma_r2), sigma_r2);
    let sigma_r12 = df64_mul(sigma_r6, sigma_r6);

    let four_eps = df64_scale_f32(eps, 4.0);
    let v = df64_mul(four_eps, df64_sub(sigma_r12, sigma_r6));
    energies[idx] = vec2<f32>(v.hi, v.lo);
}
