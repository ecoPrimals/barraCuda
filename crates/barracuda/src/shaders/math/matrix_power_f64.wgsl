// SPDX-License-Identifier: AGPL-3.0-or-later
// Matrix Power - M^n via exponentiation by squaring (f64 canonical)
// Efficiently computes matrix raised to integer power
//
// Algorithm: Exponentiation by squaring
// - M^n = M^(n/2) * M^(n/2) if n even
// - M^n = M * M^(n-1) if n odd
// - M^0 = I (identity)
//
// Note: This is a multi-pass operation requiring log(n) matrix multiplications.
// Each pass is a standard matrix multiplication.

struct Params {
    size: u32,       // Matrix size (n×n)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix_a: array<f64>;  // Left matrix
@group(0) @binding(2) var<storage, read> matrix_b: array<f64>;  // Right matrix
@group(0) @binding(3) var<storage, read_write> output: array<f64>;  // Result

// Matrix multiplication (used iteratively for power computation)
@compute @workgroup_size(16, 16, 1)
fn matrix_multiply(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.size || col >= params.size) {
        return;
    }

    var sum = 0.0;
    for (var k = 0u; k < params.size; k++) {
        sum += matrix_a[row * params.size + k] * matrix_b[k * params.size + col];
    }

    output[row * params.size + col] = sum;
}

// Initialize identity matrix
@compute @workgroup_size(16, 16, 1)
fn init_identity(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.size || col >= params.size) {
        return;
    }

    if (row == col) {
        output[row * params.size + col] = 1.0;
    } else {
        output[row * params.size + col] = 0.0;
    }
}
