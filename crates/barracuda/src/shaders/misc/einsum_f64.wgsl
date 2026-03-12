// SPDX-License-Identifier: AGPL-3.0-only
// Einsum_f64.wgsl — Einstein summation (f64 canonical)
// Flexible tensor contraction using Einstein notation
//
// Example: einsum("ij,jk->ik", A, B) is matrix multiplication

struct Params {
    output_size: u32,
    a_size: u32,
    b_size: u32,
    contract_size: u32,
    a_stride1: u32,
    a_stride2: u32,
    b_stride1: u32,
    b_stride2: u32,
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

    var sum = 0.0;

    for (var k = 0u; k < params.contract_size; k++) {
        let a_idx = (out_idx / params.b_stride2) * params.a_stride1 + k * params.a_stride2;
        let b_idx = k * params.b_stride1 + (out_idx % params.b_stride2);
        sum += tensor_a[a_idx] * tensor_b[b_idx];
    }

    output[out_idx] = sum;
}
