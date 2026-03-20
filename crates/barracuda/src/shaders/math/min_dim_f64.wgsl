// SPDX-License-Identifier: AGPL-3.0-or-later
// Min Dim: Compute minimum along a specific dimension (f64 canonical)
// Similar to PyTorch's torch.min(dim=N)
// Returns minimum values along the specified dimension

struct Params {
    dim_size: u32,
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.outer_size * params.inner_size) {
        return;
    }

    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;

    var min_value = 3.402823e+38; // FLT_MAX

    for (var i = 0u; i < params.dim_size; i++) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        let value = input[input_idx];
        min_value = min(min_value, value);
    }

    output[idx] = min_value;
}
