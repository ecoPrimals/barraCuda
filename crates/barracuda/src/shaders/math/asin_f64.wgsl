// SPDX-License-Identifier: AGPL-3.0-only
// Inverse sine (arcsine) operation (f64 canonical)
// asin(x) returns the angle whose sine is x, in range [-π/2, π/2]
// Input must be in range [-1, 1]

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }

    let x = input[idx];

    // asin is only defined for |x| <= 1
    // WGSL asin() returns NaN for out-of-range values
    output[idx] = asin_f64(x);
}
