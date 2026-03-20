// SPDX-License-Identifier: AGPL-3.0-or-later
//
// fbm_2d_f64.wgsl — Batch 2D Fractal Brownian Motion (f64)
//
// Each thread computes fBm (layered Perlin octaves) for one (x, y) pair.
// The permutation table is uploaded as a storage buffer.
//
// fBm layers multi-scale detail: octave 1 = continents, octave 2 = hills,
// octave 3+ = fine-grain texture.
//
// Absorbed from ludoSpring V2 CPU reference (Perlin 1985, 2002).
//
// Dispatch: (ceil(n_points / 256), 1, 1)

enable f64;

struct Params {
    n_points: u32,
    octaves: u32,
    _pad0: u32,
    _pad1: u32,
    lacunarity: f64,
    persistence: f64,
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

fn perlin_2d(x: f64, y: f64) -> f64 {
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

    return lerp(
        lerp(grad2(aa, xf, yf), grad2(ba, xf - 1.0, yf), u),
        lerp(grad2(ab, xf, yf - 1.0), grad2(bb, xf - 1.0, yf - 1.0), u),
        v,
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_points {
        return;
    }

    let base_x = coords[idx * 2u];
    let base_y = coords[idx * 2u + 1u];

    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;

    for (var oct = 0u; oct < params.octaves; oct = oct + 1u) {
        value = value + perlin_2d(base_x * frequency, base_y * frequency) * amplitude;
        max_value = max_value + amplitude;
        amplitude = amplitude * params.persistence;
        frequency = frequency * params.lacunarity;
    }

    output[idx] = value / max_value;
}
