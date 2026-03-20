// SPDX-License-Identifier: AGPL-3.0-or-later
// torsion_angles_df64.wgsl — phi/psi/omega backbone torsion angles from Cα coordinates
//
// Dihedral angle from 4 consecutive backbone atoms: phi = atan2(dot(n1×n2, u), dot(n1, n2))
// M = N-3 angles from quadruplets (p0,p1,p2,p3), (p1,p2,p3,p4), ...
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct Vec3Df64 {
    x: Df64,
    y: Df64,
    z: Df64,
}

struct TorsionParams {
    n: u32,
    m: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<vec2<f32>>;  // [N*3] x,y,z as DF64
@group(0) @binding(1) var<storage, read_write> out_angles: array<vec2<f32>>;  // [M] as DF64
@group(0) @binding(2) var<uniform> params: TorsionParams;

fn vec3_at(base: u32, i: u32) -> Vec3Df64 {
    let b = base + i * 3u;
    return Vec3Df64(
        Df64(positions[b].x, positions[b].y),
        Df64(positions[b + 1u].x, positions[b + 1u].y),
        Df64(positions[b + 2u].x, positions[b + 2u].y)
    );
}

fn cross_df64(a: Vec3Df64, b: Vec3Df64) -> Vec3Df64 {
    return Vec3Df64(
        df64_sub(df64_mul(a.y, b.z), df64_mul(a.z, b.y)),
        df64_sub(df64_mul(a.z, b.x), df64_mul(a.x, b.z)),
        df64_sub(df64_mul(a.x, b.y), df64_mul(a.y, b.x))
    );
}

fn dot_df64(a: Vec3Df64, b: Vec3Df64) -> Df64 {
    return df64_add(df64_add(df64_mul(a.x, b.x), df64_mul(a.y, b.y)), df64_mul(a.z, b.z));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.m { return; }

    let p0 = vec3_at(0u, idx);
    let p1 = vec3_at(0u, idx + 1u);
    let p2 = vec3_at(0u, idx + 2u);
    let p3 = vec3_at(0u, idx + 3u);

    let b1 = Vec3Df64(
        df64_sub(p1.x, p0.x), df64_sub(p1.y, p0.y), df64_sub(p1.z, p0.z)
    );
    let b2 = Vec3Df64(
        df64_sub(p2.x, p1.x), df64_sub(p2.y, p1.y), df64_sub(p2.z, p1.z)
    );
    let b3 = Vec3Df64(
        df64_sub(p3.x, p2.x), df64_sub(p3.y, p2.y), df64_sub(p3.z, p2.z)
    );

    let n1 = cross_df64(b1, b2);
    let n2 = cross_df64(b2, b3);
    let n1xn2 = cross_df64(n1, n2);

    let num = dot_df64(n1xn2, b2);
    let den = dot_df64(n1, n2);

    let angle = atan2_df64(num, den);
    out_angles[idx] = vec2<f32>(angle.hi, angle.lo);
}
