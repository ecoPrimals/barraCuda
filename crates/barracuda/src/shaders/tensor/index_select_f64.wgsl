// SPDX-License-Identifier: AGPL-3.0-only
// Index Select - Select elements by indices along a dimension (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Supports multi-dimensional tensors

struct Params {
    total_size: u32,     // Total output elements
    dim_size: u32,       // Size of indexed dimension in input
    outer_size: u32,     // Product of dims before index dim
    inner_size: u32,     // Product of dims after index dim
    num_indices: u32,    // Number of index values
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.total_size) {
        return;
    }

    // Decompose flat index into (outer, index_pos, inner)
    let inner_idx = idx % params.inner_size;
    let index_pos = (idx / params.inner_size) % params.num_indices;
    let outer_idx = idx / (params.num_indices * params.inner_size);

    // Look up the source dimension index
    let src_dim_idx = indices[index_pos];

    // Compute source flat index
    let src_idx = outer_idx * (params.dim_size * params.inner_size)
                + src_dim_idx * params.inner_size
                + inner_idx;

    output[idx] = input[src_idx];
}
