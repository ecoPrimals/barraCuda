// SPDX-License-Identifier: AGPL-3.0-or-later
// sinh_f64.wgsl — Hyperbolic sine operation (f64 canonical)
// sinh(x) = (e^x - e^(-x)) / 2

struct Metadata {
    size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }

    let x = input[idx];
    output[idx] = sinh_f64(x);
}
