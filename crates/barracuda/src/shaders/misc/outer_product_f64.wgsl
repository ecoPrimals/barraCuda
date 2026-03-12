// SPDX-License-Identifier: AGPL-3.0-only
// Outer Product - Tensor product of vectors (complete parallel implementation) (f64 canonical)
// Creates matrix M[i,j] = a[i] * b[j]
//
// Algorithm:
// Each thread computes one element of the output matrix

struct Params {
    size_a: u32,     // Length of vector a
    size_b: u32,     // Length of vector b
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> vec_a: array<f64>;
@group(0) @binding(2) var<storage, read> vec_b: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;  // [size_a, size_b]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;
    let j = global_id.x;
    
    if (i >= params.size_a || j >= params.size_b) {
        return;
    }

    output[i * params.size_b + j] = vec_a[i] * vec_b[j];
}
