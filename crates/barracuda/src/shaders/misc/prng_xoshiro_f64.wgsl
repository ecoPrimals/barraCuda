// SPDX-License-Identifier: AGPL-3.0-only
//
// prng_xoshiro_f64.wgsl — Xoshiro128** PRNG with f64 output
//
// Same PRNG algorithm (u32 state), but output is uniform f64 in [0, 1).
// Uses two u32 draws combined for full f64 mantissa coverage (52 bits).
//
// Evolved from f32 → f64 for universal math library portability.

@group(0) @binding(0) var<storage, read> seeds: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    offset: u32,
}

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn splitmix32(x: u32) -> u32 {
    var z = x;
    z = (z ^ (z >> 16u)) * 0x85ebca6bu;
    z = (z ^ (z >> 13u)) * 0xc2b2ae35u;
    return z ^ (z >> 16u);
}

fn xoshiro_next(s: ptr<function, array<u32, 4>>) -> u32 {
    let result = rotl((*s)[1] * 5u, 7u) * 9u;
    let t = (*s)[1] << 9u;
    (*s)[2] = (*s)[2] ^ (*s)[0];
    (*s)[3] = (*s)[3] ^ (*s)[1];
    (*s)[1] = (*s)[1] ^ (*s)[2];
    (*s)[0] = (*s)[0] ^ (*s)[3];
    (*s)[2] = (*s)[2] ^ t;
    (*s)[3] = rotl((*s)[3], 11u);
    return result;
}

fn to_uniform_f64(s: ptr<function, array<u32, 4>>) -> f64 {
    // Combine two u32 draws for 52-bit mantissa coverage
    let hi = xoshiro_next(s);
    let lo = xoshiro_next(s);
    // hi provides upper 26 bits, lo provides lower 26 bits
    let combined = f64(hi >> 6u) * f64(67108864.0) + f64(lo >> 6u);
    return combined / f64(9007199254740992.0); // 2^53
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let seed_base = idx * 4u;
    var state: array<u32, 4>;
    var z = seeds[seed_base];
    state[0] = splitmix32(z);
    z = z + 0x9e3779b9u;
    state[1] = splitmix32(z);
    z = z + 0x9e3779b9u;
    state[2] = splitmix32(z);
    z = z + 0x9e3779b9u;
    state[3] = splitmix32(z);

    if (state[0] == 0u && state[1] == 0u && state[2] == 0u && state[3] == 0u) {
        state[0] = 1u;
    }

    for (var i = 0u; i < params.offset; i = i + 1u) {
        let _unused = xoshiro_next(&state);
    }

    output[idx] = to_uniform_f64(&state);
}
