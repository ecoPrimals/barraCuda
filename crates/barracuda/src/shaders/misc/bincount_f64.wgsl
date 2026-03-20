// SPDX-License-Identifier: AGPL-3.0-or-later
// Bincount_f64.wgsl — Count occurrences of each value (f64 canonical)
// Computes histogram of non-negative integer values

struct Params {
    size: u32,
    num_bins: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let value = u32(trunc(input[idx]));

    // Only count if value is within valid bin range
    if (value < params.num_bins) {
        atomicAdd(&output[value], 1u);
    }
}
