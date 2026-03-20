// SPDX-License-Identifier: AGPL-3.0-or-later
// Triu - Upper triangular matrix (complete parallel implementation) - f64 canonical
// Zeros elements below diagonal
//
// Example: triu([[1,2,3],[4,5,6],[7,8,9]], diagonal=0) → [[1,2,3],[0,5,6],[0,0,9]]
//
// Algorithm:
// Each thread checks if its position is above/on diagonal and zeroes if below

struct Params {
    rows: u32,
    cols: u32,
    diagonal: i32,  // Offset: 0=main diagonal, 1=above, -1=below
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    
    if (row >= params.rows || col >= params.cols) {
        return;
    }

    let idx = row * params.cols + col;
    
    // Check if position is above diagonal (including offset)
    if (i32(col) >= i32(row) + params.diagonal) {
        output[idx] = input[idx];
    } else {
        output[idx] = f64(0.0);
    }
}
