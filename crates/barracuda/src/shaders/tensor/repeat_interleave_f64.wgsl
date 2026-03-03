// Repeat Interleave - Repeat each element along dimension (f64 canonical)
// Repeats each element `repeats` times along specified dimension
//
// Example: repeat_interleave([1, 2, 3], 2, dim=0) → [1, 1, 2, 2, 3, 3]
//
// Algorithm:
// For each output position, determine which input element to copy

struct Params {
    output_size: u32,
    input_size: u32,
    repeats: u32,
    dim: u32,
    dim_size: u32,       // Size of dimension being repeated
    inner_size: u32,     // Product of dimensions after dim
    outer_size: u32,     // Product of dimensions before dim
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.output_size) {
        return;
    }

    // Decompose output index
    let outer = out_idx / (params.dim_size * params.repeats * params.inner_size);
    let temp = out_idx % (params.dim_size * params.repeats * params.inner_size);
    let dim_idx_repeated = temp / params.inner_size;
    let inner = temp % params.inner_size;

    // Map repeated dimension index back to original
    let dim_idx_original = dim_idx_repeated / params.repeats;

    // Compute input index
    let in_idx = outer * params.dim_size * params.inner_size
                 + dim_idx_original * params.inner_size
                 + inner;

    output[out_idx] = input[in_idx];
}
