// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Shared PRNG core: Wang hash → xorshift32 → uniform f64 on [0, 1).
//
// Consumer shaders include this via Rust string concatenation (same
// pattern as lattice/prng_pcg_f64.wgsl). The consumer defines its
// own Params struct + bindings before this library.
//
// Absorbs the duplicated PRNG from healthSpring (V19–V44) into a
// single source of truth.

/// Thomas Wang integer hash (2007). Produces well-distributed u32
/// from sequential seeds — avalanche property ensures adjacent
/// patient indices yield uncorrelated streams.
fn wang_hash(input: u32) -> u32 {
    var x = input;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x * 9u;
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

/// Marsaglia xorshift32 — period 2^32 - 1. State must be non-zero.
fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

/// Map u32 uniformly to [0, 1) via division by 2^32 - 1.
fn u32_to_uniform_f64(bits: u32) -> f64 {
    return f64(bits) / 4294967295.0;   // (2^32 - 1)
}
