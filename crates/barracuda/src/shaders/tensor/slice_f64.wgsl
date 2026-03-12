// SPDX-License-Identifier: AGPL-3.0-only
// Slice operation - extract subtensor (f64 canonical)
// Parameters: start index, length

struct SliceParams {
    start: u32,
    length: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: SliceParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.length) {
        return;
    }
    
    let src_idx = params.start + idx;
    if (src_idx < arrayLength(&input)) {
        output[idx] = input[src_idx];
    }
}
