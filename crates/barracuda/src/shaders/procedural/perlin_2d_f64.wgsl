// SPDX-License-Identifier: AGPL-3.0-only
//
// perlin_2d_f64.wgsl — Batch 2D Perlin noise (f64)
//
// Each thread computes Perlin noise for one (x, y) coordinate pair.
// The 256-entry permutation table is uploaded as a storage buffer.
//
// Absorbed from ludoSpring V2 CPU reference (Perlin 1985, 2002; Gustavson 2005).
//
// Dispatch: (ceil(n_points / 256), 1, 1)

enable f64;

struct Params {
    n_points: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> coords: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<storage, read> perm: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

fn fade(t: f64) -> f64 {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    return a + t * (b - a);
}

fn grad2(hash: u32, x: f64, y: f64) -> f64 {
    let h = hash & 3u;
    if h == 0u { return x + y; }
    if h == 1u { return -x + y; }
    if h == 2u { return x - y; }
    return -x - y;
}

fn perm_lookup(i: u32) -> u32 {
    return perm[i & 255u];
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_points {
        return;
    }

    let x = coords[idx * 2u];
    let y = coords[idx * 2u + 1u];

    let xi = u32(floor(x)) & 255u;
    let yi = u32(floor(y)) & 255u;
    let xf = x - floor(x);
    let yf = y - floor(y);

    let u = fade(xf);
    let v = fade(yf);

    let aa = perm_lookup(perm_lookup(xi) + yi);
    let ab = perm_lookup(perm_lookup(xi) + yi + 1u);
    let ba = perm_lookup(perm_lookup(xi + 1u) + yi);
    let bb = perm_lookup(perm_lookup(xi + 1u) + yi + 1u);

    let result = lerp(
        lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u),
        lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u),
        v,
    );

    output[idx] = result;
}
