// SPDX-License-Identifier: AGPL-3.0-only
// Variance Dim (f64) - Variance along a specific dimension at double precision
//
// Similar to PyTorch's torch.var(dim=N)
// Returns variance values along the specified dimension
// Var(X) = E[(X - μ)²] where μ is the mean along the dimension

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

    // First pass: compute mean along the dimension
    var sum_value = f64(0.0);

    for (var i = 0u; i < params.dim_size; i++) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        let value = input[input_idx];
        sum_value = sum_value + value;
    }

    let mean = sum_value / f64(params.dim_size);

    // Second pass: compute variance along the dimension
    var variance = f64(0.0);

    for (var i = 0u; i < params.dim_size; i++) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        let value = input[input_idx];
        let diff = value - mean;
        variance = variance + diff * diff;
    }

    output[idx] = variance / f64(params.dim_size);
}
