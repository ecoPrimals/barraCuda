// SPDX-License-Identifier: AGPL-3.0-only
// hydrogen_bond_df64.wgsl — Hydrogen bond energy (donor-acceptor geometry)
//
// Simplified 12-10-6 potential: V = eps * (5*(sigma/r)^12 - 6*(sigma/r)^10)
// Or distance-angle: V = A/r^12 - B/r^10 for D-H...A geometry.
// Per H-bond pair. Input: D-A distance, optional angle factor.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct HbondParams {
    n_pairs: u32,
    sigma: f32,
    epsilon: f32,
}

@group(0) @binding(0) var<storage, read> distances: array<vec2<f32>>;  // [M] D-A distances
@group(0) @binding(1) var<storage, read_write> energies: array<vec2<f32>>;  // [M] H-bond energy
@group(0) @binding(2) var<uniform> params: HbondParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_pairs { return; }

    let r = Df64(distances[idx].x, distances[idx].y);
    let sigma = df64_from_f32(params.sigma);
    let eps = df64_from_f32(params.epsilon);

    let r_min = df64_from_f32(0.1);
    var r_safe = r;
    if df64_lt(r, r_min) {
        r_safe = r_min;
    }

    let sigma_r = df64_div(sigma, r_safe);
    let sigma_r5 = df64_mul(df64_mul(sigma_r, sigma_r), df64_mul(sigma_r, df64_mul(sigma_r, sigma_r)));
    let sigma_r10 = df64_mul(sigma_r5, sigma_r5);
    let sigma_r12 = df64_mul(sigma_r10, df64_mul(sigma_r, sigma_r));

    // 12-10 potential: V = eps * (5*(s/r)^12 - 6*(s/r)^10)
    let five = df64_from_f32(5.0);
    let six = df64_from_f32(6.0);
    let v = df64_mul(eps, df64_sub(df64_mul(five, sigma_r12), df64_mul(six, sigma_r10)));
    energies[idx] = vec2<f32>(v.hi, v.lo);
}
