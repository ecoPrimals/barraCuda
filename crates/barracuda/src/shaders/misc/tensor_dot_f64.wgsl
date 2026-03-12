// SPDX-License-Identifier: AGPL-3.0-only
// Tensor Dot - Generalized tensor contraction (complete implementation) (f64 canonical)
// Performs tensor dot product over specified axes
//
// Example: tensordot(A[i,j,k], B[k,l,m], axes=[[2], [0]]) → C[i,j,l,m]
//
// Algorithm:
// 1. Contract over specified axes (sum-product)
// 2. Output has remaining uncontracted dimensions

struct Params {
    output_size: u32,
    contraction_size: u32,   // Product of contracted dimensions
    a_outer_size: u32,       // Product of A's uncontracted dimensions
    b_outer_size: u32,       // Product of B's uncontracted dimensions
    a_outer_stride: u32,     // Stride for A's outer dimensions
    b_outer_stride: u32,     // Stride for B's outer dimensions
    a_contract_stride: u32,  // Stride for A's contraction dimensions
    b_contract_stride: u32,  // Stride for B's contraction dimensions
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tensor_a: array<f64>;
@group(0) @binding(2) var<storage, read> tensor_b: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_idx = global_id.x;
    if (out_idx >= params.output_size) {
        return;
    }

    // Decompose output index into A's outer index and B's outer index
    let a_outer_idx = out_idx / params.b_outer_size;
    let b_outer_idx = out_idx % params.b_outer_size;
    
    // Contract over shared dimensions
    var sum = 0.0;
    for (var k = 0u; k < params.contraction_size; k++) {
        let a_idx = a_outer_idx * params.a_contract_stride + k;
        let b_idx = k * params.b_outer_stride + b_outer_idx;
        sum += tensor_a[a_idx] * tensor_b[b_idx];
    }
    
    output[out_idx] = sum;
}
