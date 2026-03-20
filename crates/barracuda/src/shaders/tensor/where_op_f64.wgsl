// SPDX-License-Identifier: AGPL-3.0-or-later
// Where - Conditional selection (complete parallel implementation) (f64 canonical)
// Selects from two tensors based on condition mask
//
// Example: where(condition, x, y) → x if condition else y
//
// Algorithm:
// Each thread selects from x or y based on its condition value

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> condition: array<u32>;  // 0 or 1 (boolean)
@group(0) @binding(2) var<storage, read> x: array<f64>;
@group(0) @binding(3) var<storage, read> y: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    if (condition[idx] != 0u) {
        output[idx] = x[idx];
    } else {
        output[idx] = y[idx];
    }
}
