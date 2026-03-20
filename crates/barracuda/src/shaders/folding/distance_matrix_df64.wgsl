// SPDX-License-Identifier: AGPL-3.0-or-later
// distance_matrix_df64.wgsl — pairwise Cα distance matrix
//
// Computes d[i,j] = ||pos[i] - pos[j]|| for all residue pairs.
// Output: N×N distance matrix, stored row-major.
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct Vec3Df64 {
    x: Df64,
    y: Df64,
    z: Df64,
}

struct DistMatrixParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;  // [N*3] x,y,z as DF64
@group(0) @binding(1) var<storage, read_write> dist_matrix: array<vec2<f32>>;  // [N*N] as DF64
@group(0) @binding(2) var<uniform> params: DistMatrixParams;

fn vec3_at(base: u32, i: u32) -> Vec3Df64 {
    let b = base + i * 3u;
    return Vec3Df64(
        Df64(positions[b].x, positions[b].y),
        Df64(positions[b + 1u].x, positions[b + 1u].y),
        Df64(positions[b + 2u].x, positions[b + 2u].y)
    );
}

fn dist_sq_df64(a: Vec3Df64, b: Vec3Df64) -> Df64 {
    let dx = df64_sub(a.x, b.x);
    let dy = df64_sub(a.y, b.y);
    let dz = df64_sub(a.z, b.z);
    return df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let N = params.n;
    if idx >= N * N { return; }

    let i = idx / N;
    let j = idx % N;

    let pi = vec3_at(0u, i);
    let pj = vec3_at(0u, j);

    let d_sq = dist_sq_df64(pi, pj);
    let d = sqrt_df64(d_sq);
    dist_matrix[idx] = vec2<f32>(d.hi, d.lo);
}
