// SPDX-License-Identifier: AGPL-3.0-or-later
// Histc_f64.wgsl — Histogram with custom bins (f64 canonical)
// Computes histogram of input values into specified bins
// Uses atomic operations for parallel histogram computation

struct Params {
    size: u32,
    num_bins: u32,
    min_val: f64,
    max_val: f64,
    bin_width: f64,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> histogram: array<atomic<u32>>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let val = input[idx];

    if (val >= params.min_val && val < params.max_val) {
        let bin_idx = u32(floor((val - params.min_val) / params.bin_width));
        let bin = min(bin_idx, params.num_bins - 1u);
        atomicAdd(&histogram[bin], 1u);
    }
}
