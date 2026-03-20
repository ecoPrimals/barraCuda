// SPDX-License-Identifier: AGPL-3.0-or-later
// contact_map_df64.wgsl — Binary contact map from distance matrix with threshold
//
// contact[i,j] = 1 if dist[i,j] <= threshold else 0
// Input: distance matrix [N*N] as DF64. Output: binary [N*N] as f32 (0 or 1).
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct ContactParams {
    n: u32,
    threshold: f32,  // distance threshold in Angstroms
}

@group(0) @binding(0) var<storage, read> dist_matrix: array<vec2<f32>>;  // [N*N] as DF64
@group(0) @binding(1) var<storage, read_write> contact_map: array<f32>;  // [N*N] binary
@group(0) @binding(2) var<uniform> params: ContactParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n * params.n { return; }

    let d = Df64(dist_matrix[idx].x, dist_matrix[idx].y);
    let thresh = df64_from_f32(params.threshold);
    let is_contact = !df64_gt(d, thresh);  // d <= thresh
    contact_map[idx] = select(0.0, 1.0, is_contact);
}
