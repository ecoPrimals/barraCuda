// SPDX-License-Identifier: AGPL-3.0-only
// Chunk - Split tensor into N chunks along dimension (complete implementation) (f64 canonical)
// Divides input tensor into chunks (supports unequal chunks like PyTorch)
// When dimension size is not divisible by chunks:
//   - First (dim_size % chunks) chunks get (dim_size // chunks) + 1 elements
//   - Remaining chunks get (dim_size // chunks) elements
//
// Example: chunk([B, 10, H, W], chunks=3, dim=1) → [[B, 4, H, W], [B, 4, H, W], [B, 2, H, W]]
//
// Algorithm:
// For each chunk, copy the appropriate slice of the input tensor using start_offset

struct Params {
    start_offset: u32,  // Start offset in the split dimension for this chunk
    chunk_size: u32,     // Size of this chunk along split dimension
    split_dim: u32,      // Dimension to split
    dim_size: u32,       // Size of dimension being split
    inner_size: u32,     // Product of dimensions after split_dim
    outer_size: u32,     // Product of dimensions before split_dim
    output_size: u32,    // Total output size for this chunk
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
    let outer = out_idx / (params.chunk_size * params.inner_size);
    let temp = out_idx % (params.chunk_size * params.inner_size);
    let chunk_coord = temp / params.inner_size;
    let inner = temp % params.inner_size;

    // Map to input index
    let input_coord = params.start_offset + chunk_coord;
    let in_idx = outer * params.dim_size * params.inner_size
                 + input_coord * params.inner_size
                 + inner;

    output[out_idx] = input[in_idx];
}
