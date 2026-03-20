// SPDX-License-Identifier: AGPL-3.0-or-later
// rsqrt_f64.wgsl — Element-wise reciprocal square root (1/sqrt(x)) (f64 canonical)
//
// Universal compute via WGSL - works on any hardware.
// Uses 1/sqrt_f64(x) for f64 canonical; sqrt is native on f64 hardware.

struct Params {
    size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Compute reciprocal square root: 1/sqrt(x)
    output[idx] = 1.0 / sqrt_f64(input[idx]);
}
