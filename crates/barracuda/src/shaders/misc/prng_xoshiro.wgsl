// SPDX-License-Identifier: AGPL-3.0-or-later
// PRNG xoshiro128** - High-quality 32-bit pseudorandom number generator
// Adapted for WGSL from the algorithm by David Blackman and Sebastiano Vigna
// Each thread generates one random value from its seed; seeds passed as f32 (bitcast from u32)

@group(0) @binding(0) var<storage, read> seeds: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    offset: u32,
}

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

// SplitMix32: derive 32-bit hash from seed (used for state initialization)
fn splitmix32(x: u32) -> u32 {
    var z = x;
    z = (z ^ (z >> 16u)) * 0x85ebca6bu;
    z = (z ^ (z >> 13u)) * 0xc2b2ae35u;
    return z ^ (z >> 16u);
}

// xoshiro128** next: advance state and return random u32
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

// Convert u32 to uniform f32 in [0, 1)
fn to_uniform_01(bits: u32) -> f32 {
    // Use top 23 bits for mantissa; 0x3f800000 = 1.0 in float
    return bitcast<f32>((bits >> 9u) | 0x3f800000u) - 1.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    // Read seed as u32 (stored as f32 bitcast)
    let seed_bits = bitcast<u32>(seeds[idx]);

    // Initialize state via SplitMix32 (ensure non-zero)
    var state: array<u32, 4>;
    var z = seed_bits;
    state[0] = splitmix32(z);
    z = z + 0x9e3779b9u;
    state[1] = splitmix32(z);
    z = z + 0x9e3779b9u;
    state[2] = splitmix32(z);
    z = z + 0x9e3779b9u;
    state[3] = splitmix32(z);

    // Ensure state is not all zeros
    if (state[0] == 0u && state[1] == 0u && state[2] == 0u && state[3] == 0u) {
        state[0] = 1u;
    }

    // Advance by offset steps
    for (var i = 0u; i < params.offset; i = i + 1u) {
        let _unused = xoshiro_next(&state);
    }

    // Generate one random value
    let bits = xoshiro_next(&state);
    output[idx] = to_uniform_01(bits);
}
