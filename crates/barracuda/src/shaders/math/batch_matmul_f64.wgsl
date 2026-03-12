// SPDX-License-Identifier: AGPL-3.0-only
// BatchMatMul - Batched matrix multiplication (f64 canonical)
// Critical for transformer attention: (batch, heads, seq, seq) matrix products
// More efficient than looping over MatMul for batched operations

struct BatchMatMulParams {
    batch_size: u32,
    m: u32,  // rows of A
    n: u32,  // cols of B
    k: u32,  // cols of A / rows of B
}

@group(0) @binding(0) var<storage, read> a: array<f64>;  // [batch, m, k]
@group(0) @binding(1) var<storage, read> b: array<f64>;  // [batch, k, n]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // [batch, m, n]
@group(0) @binding(3) var<uniform> params: BatchMatMulParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let batch = global_id.z;

    if (row >= params.m || col >= params.n || batch >= params.batch_size) {
        return;
    }

    var sum = 0.0;
    let matrix_size_a = params.m * params.k;
    let matrix_size_b = params.k * params.n;

    // Compute dot product for this position
    for (var i = 0u; i < params.k; i = i + 1u) {
        let a_idx = batch * matrix_size_a + row * params.k + i;
        let b_idx = batch * matrix_size_b + i * params.n + col;
        sum += a[a_idx] * b[b_idx];
    }

    let out_idx = batch * params.m * params.n + row * params.n + col;
    output[out_idx] = sum;
}
