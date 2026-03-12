// SPDX-License-Identifier: AGPL-3.0-only
// structure_module_df64.wgsl — IPA (Invariant Point Attention) frame update
//
// IPA score: q*k^T/sqrt(d) + w_l * ||T_i(q_pts) - T_j(k_pts)||^2
// Scalar attention for (i,j). Simplified: identity transforms.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct IpaParams {
    n: u32,
    d: u32,
    p: u32,
    w_l: f32,
}

@group(0) @binding(0) var<storage, read> q: array<vec2<f32>>;  // [N*D]
@group(0) @binding(1) var<storage, read> k: array<vec2<f32>>;  // [N*D]
@group(0) @binding(2) var<storage, read> q_pts: array<vec2<f32>>;  // [N*P*3]
@group(0) @binding(3) var<storage, read> k_pts: array<vec2<f32>>;  // [N*P*3]
@group(0) @binding(4) var<storage, read_write> out: array<vec2<f32>>;  // [N*N]
@group(0) @binding(5) var<uniform> params: IpaParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n;
    let D = params.d;
    let P = params.p;
    let w_l = df64_from_f32(params.w_l);
    let idx = gid.x;
    if idx >= N * N { return; }

    let i = idx / N;
    let j = idx % N;

    var dot = df64_from_f32(0.0);
    let q_base = i * D;
    let k_base = j * D;
    for (var d = 0u; d < D; d = d + 1u) {
        let qv = Df64(q[q_base + d].x, q[q_base + d].y);
        let kv = Df64(k[k_base + d].x, k[k_base + d].y);
        dot = df64_add(dot, df64_mul(qv, kv));
    }
    let scale = sqrt_df64(df64_from_f32(f32(D)));
    var score = df64_div(dot, scale);

    var geom_sq = df64_from_f32(0.0);
    let q_pts_base = i * P * 3u;
    let k_pts_base = j * P * 3u;
    for (var p = 0u; p < P; p = p + 1u) {
        let qx = Df64(q_pts[q_pts_base + p * 3u + 0u].x, q_pts[q_pts_base + p * 3u + 0u].y);
        let qy = Df64(q_pts[q_pts_base + p * 3u + 1u].x, q_pts[q_pts_base + p * 3u + 1u].y);
        let qz = Df64(q_pts[q_pts_base + p * 3u + 2u].x, q_pts[q_pts_base + p * 3u + 2u].y);
        let kx = Df64(k_pts[k_pts_base + p * 3u + 0u].x, k_pts[k_pts_base + p * 3u + 0u].y);
        let ky = Df64(k_pts[k_pts_base + p * 3u + 1u].x, k_pts[k_pts_base + p * 3u + 1u].y);
        let kz = Df64(k_pts[k_pts_base + p * 3u + 2u].x, k_pts[k_pts_base + p * 3u + 2u].y);
        let dx = df64_sub(qx, kx);
        let dy = df64_sub(qy, ky);
        let dz = df64_sub(qz, kz);
        geom_sq = df64_add(geom_sq, df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz)));
    }
    score = df64_add(score, df64_mul(w_l, geom_sq));

    out[idx] = vec2<f32>(score.hi, score.lo);
}
