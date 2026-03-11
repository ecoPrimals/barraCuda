// SPDX-License-Identifier: AGPL-3.0-only
//
// torsion_angles_f64.wgsl — Dihedral angle from 4 backbone atom positions (AlphaFold2 structure)
//
// phi = atan2(dot(n1 x n2, u), dot(n1, n2))
// M = N-3 angles from consecutive quadruplets (p0,p1,p2,p3), (p1,p2,p3,p4), ...
//
// Bindings: @0 positions[N*3], @1 out_angles[M], @2 uniform{n: u32, m: u32}
//
// Provenance: neuralSpring → ToadStool absorption

enable f64;

struct TorsionParams {
    n: u32,
    m: u32,
}

@group(0) @binding(0) var<storage, read>       positions: array<f64>;  // [N*3] flattened xyz
@group(0) @binding(1) var<storage, read_write> out_angles: array<f64>;  // [M]
@group(0) @binding(2) var<uniform>             params:     TorsionParams;

fn vec3_at(base: u32, i: u32) -> vec3<f64> {
    let b = base + i * 3u;
    return vec3<f64>(positions[b], positions[b + 1u], positions[b + 2u]);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.m { return; }

    let p0 = vec3_at(0u, idx);
    let p1 = vec3_at(0u, idx + 1u);
    let p2 = vec3_at(0u, idx + 2u);
    let p3 = vec3_at(0u, idx + 3u);

    let b1 = p1 - p0;
    let b2 = p2 - p1;
    let b3 = p3 - p2;

    let n1 = cross(b1, b2);
    let n2 = cross(b2, b3);

    let n1xn2 = cross(n1, n2);
    let u = b2;

    let num = dot(n1xn2, u);
    let den = dot(n1, n2);

    out_angles[idx] = atan2_f64(num, den);
}
