// SPDX-License-Identifier: AGPL-3.0-only
// Boltzmann Sampling (f64) — wateringHole V69
//
// GPU-accelerated Boltzmann (softmax) sampling with temperature.
// Gumbel-max: sample = argmax(logits/temp + Gumbel)
//
// Input: logits array [batch, n_classes], temperature in params (bitcast from u32 pair)
// Output: sampled indices (one per batch element)
//
// f64 enabled by compile_shader_f64() preamble injection
enable f64;

struct Params {
    batch_size: u32,
    n_classes: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f64>;
@group(0) @binding(1) var<storage, read_write> seeds: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read> temp_buf: array<f64>;

fn rotl(x: u32, k: u32) -> u32 {
    return (x << k) | (x >> (32u - k));
}

fn xoshiro_next(s: ptr<function, vec4<u32>>) -> u32 {
    let result = rotl((*s).y * 5u, 7u) * 9u;
    let t = (*s).y << 9u;
    (*s).z ^= (*s).x;
    (*s).w ^= (*s).y;
    (*s).y ^= (*s).z;
    (*s).x ^= (*s).w;
    (*s).z ^= t;
    (*s).w = rotl((*s).w, 11u);
    return result;
}

fn xoshiro_next_f64(s: ptr<function, vec4<u32>>) -> f64 {
    let hi = xoshiro_next(s);
    let lo = xoshiro_next(s);
    let combined = (f64(hi) * 4294967296.0 + f64(lo));
    return combined / 18446744073709551616.0;
}

// Inline log approximation to avoid polyfill injection (avoids vec2<f64> type confusion).
// log(x) for x in (0.5, 1] via Taylor: log(x) ≈ (x-1) - (x-1)²/2 + (x-1)³/3
fn log_approx(x: f64) -> f64 {
    if (x <= 0.0) { return f64(-1e30); }
    var v = x;
    var add = f64(0.0);
    if (v > 1.0) {
        v = 1.0 / v;
        add = f64(-1.0);
    }
    if (v < 0.5) {
        v = 1.0 / v;
        add = f64(1.0);
    }
    let t = v - 1.0;
    let t2 = t * t;
    let t3 = t2 * t;
    return add - t + t2 * f64(0.5) - t3 / f64(3.0);
}

// Gumbel(0,1) = -log(-log(U))
fn gumbel(s: ptr<function, vec4<u32>>) -> f64 {
    let u = xoshiro_next_f64(s);
    let u_clamped = clamp(u, 1e-15, 1.0 - 1e-15);
    return -log_approx(-log_approx(u_clamped));
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_idx = gid.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let n = params.n_classes;
    let base = batch_idx * n;
    let seed_base = batch_idx * 4u;

    var state = vec4<u32>(
        seeds[seed_base],
        seeds[seed_base + 1u],
        seeds[seed_base + 2u],
        seeds[seed_base + 3u],
    );

    let temp = temp_buf[0];
    let temp_safe = select(f64(1.0), temp, temp > f64(1e-10));
    var max_val: f64 = logits[base] / temp_safe + gumbel(&state);
    var argmax: u32 = 0u;

    for (var i: u32 = 1u; i < n; i = i + 1u) {
        let val = logits[base + i] / temp_safe + gumbel(&state);
        if (val > max_val) {
            max_val = val;
            argmax = i;
        }
    }

    output[batch_idx] = argmax;

    seeds[seed_base] = state.x;
    seeds[seed_base + 1u] = state.y;
    seeds[seed_base + 2u] = state.z;
    seeds[seed_base + 3u] = state.w;
}
