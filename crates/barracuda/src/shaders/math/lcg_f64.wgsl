// SPDX-License-Identifier: AGPL-3.0-or-later
// lcg_f64.wgsl — GPU PRNG for lattice kernels (xorshift32 + Box-Muller)
//
// Prepend: (standalone, no dependencies)
//
// Uses xorshift32 for speed and u32-only operation (no SHADER_INT64 needed).
// State is a single u32 per thread. Box-Muller for Gaussian samples.

fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

fn prng_uniform(state: ptr<function, u32>) -> f64 {
    let bits = xorshift32(state);
    return f64(bits) / f64(4294967295.0);
}

fn prng_gaussian(state: ptr<function, u32>) -> f64 {
    var u1 = prng_uniform(state);
    if (u1 < f64(1e-30)) { u1 = f64(1e-30); }
    let u2 = prng_uniform(state);
    let two_pi: f64 = f64(6.283185307179586);
    return sqrt(f64(-2.0) * log(u1)) * cos(two_pi * u2);
}
