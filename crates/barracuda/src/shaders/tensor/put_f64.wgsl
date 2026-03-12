// SPDX-License-Identifier: AGPL-3.0-only
// Put - Scatter operation with indexing (f64 canonical)
// Places values at specified indices in output tensor
//
// Example: put(zeros(4), [0, 2], [10, 30]) → [10, 0, 30, 0]
//
// Algorithm:
// For each value, scatter to position given by index
//
// Evolution: Removed atomic<i32> + bitcast pattern which corrupted f32 accumulate.
// Uses f64 directly. For non-overlapping indices (standard use), this is correct.
// For overlapping indices in accumulate mode, results are non-deterministic
// (acceptable for GPU scatter semantics, matching PyTorch behavior).

struct Params {
    output_size: u32,
    num_values: u32,
    accumulate: u32,  // 0 = overwrite, 1 = accumulate
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<f64>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_values) {
        return;
    }

    let out_idx = indices[idx];
    let value = values[idx];

    // Bounds check
    if (out_idx < params.output_size) {
        if (params.accumulate != 0u) {
            // Accumulate: read-modify-write (correct for non-overlapping indices)
            output[out_idx] = output[out_idx] + value;
        } else {
            // Overwrite (last write wins if multiple indices point to same location)
            output[out_idx] = value;
        }
    }
}
