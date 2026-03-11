// SPDX-License-Identifier: AGPL-3.0-only
//
// ipa_scores_f64.wgsl — Invariant Point Attention scores (AlphaFold2)
//
// Computes: q*k^T/sqrt(d) + w_l * ||T_i(q_pts) - T_j(k_pts)||^2
// Scalar attention score for each (i,j) pair. The geometric term uses squared L2 distance
// between transformed point clouds. For simplicity we assume T_i, T_j are identity here;
// external code may apply transforms before passing q_pts, k_pts.
//
// Bindings: @0 q[N*D], @1 k[N*D], @2 q_pts[N*P*3], @3 k_pts[N*P*3], @4 out[N*N], @5 uniform{n, d, p, w_l}

enable f64;

struct IpaParams {
    n: u32,
    d: u32,
    p: u32,
    w_l: f64,
}

@group(0) @binding(0) var<storage, read>       q: array<f64>;       // [N*D]
@group(0) @binding(1) var<storage, read>       k: array<f64>;       // [N*D]
@group(0) @binding(2) var<storage, read>       q_pts: array<f64>;   // [N*P*3]
@group(0) @binding(3) var<storage, read>       k_pts: array<f64>;   // [N*P*3]
@group(0) @binding(4) var<storage, read_write> out: array<f64>;     // [N*N]
@group(0) @binding(5) var<uniform>             params: IpaParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n;
    let D = params.d;
    let P = params.p;
    let w_l = params.w_l;
    let idx = gid.x;
    if idx >= N * N { return; }

    let i = idx / N;
    let j = idx % N;

    var dot = f64(0.0);
    let q_base = i * D;
    let k_base = j * D;
    for (var d = 0u; d < D; d = d + 1u) {
        dot += q[q_base + d] * k[k_base + d];
    }
    let scale = sqrt_f64(f64(D));
    var score = dot / scale;

    var geom_sq = f64(0.0);
    let q_pts_base = i * P * 3u;
    let k_pts_base = j * P * 3u;
    for (var p = 0u; p < P; p = p + 1u) {
        let qx = q_pts[q_pts_base + p * 3u + 0u];
        let qy = q_pts[q_pts_base + p * 3u + 1u];
        let qz = q_pts[q_pts_base + p * 3u + 2u];
        let kx = k_pts[k_pts_base + p * 3u + 0u];
        let ky = k_pts[k_pts_base + p * 3u + 1u];
        let kz = k_pts[k_pts_base + p * 3u + 2u];
        let dx = qx - kx;
        let dy = qy - ky;
        let dz = qz - kz;
        geom_sq += dx * dx + dy * dy + dz * dz;
    }
    score += w_l * geom_sq;

    out[idx] = score;
}
