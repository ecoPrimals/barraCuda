// SPDX-License-Identifier: AGPL-3.0-or-later
// MatMul - Matrix multiplication (f64 canonical)
// C = A * B
// Simplified version: 2D matrix multiplication

struct MatMulParams {
    m: u32,  // rows of A
    k: u32,  // cols of A / rows of B
    n: u32,  // cols of B
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> c: array<f64>;
@group(0) @binding(3) var<uniform> params: MatMulParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= params.m || col >= params.n) {
        return;
    }

    var sum = 0.0;
    for (var i = 0u; i < params.k; i = i + 1u) {
        let a_idx = row * params.k + i;
        let b_idx = i * params.n + col;
        sum = sum + a[a_idx] * b[b_idx];
    }

    let c_idx = row * params.n + col;
    c[c_idx] = sum;
}
