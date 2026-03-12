// SPDX-License-Identifier: AGPL-3.0-only
// RReLU - Randomized Leaky ReLU (f64 canonical)
// output = x if x > 0 else slope*x where slope ~ Uniform(lower, upper)

struct Params {
    size: u32,
    lower: f64,
    upper: f64,
    seed: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

fn lcg(seed: u32) -> u32 {
    return (1103515245u * seed + 12345u) & 0x7fffffffu;
}

fn random_f64(seed: u32) -> f64 {
    return f64(lcg(seed)) / f64(0x7fffffffu);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];

    if (x > 0.0) {
        output[idx] = x;
    } else {
        let seed = params.seed + idx;
        let rand = random_f64(seed);
        let slope = params.lower + rand * (params.upper - params.lower);
        output[idx] = slope * x;
    }
}
