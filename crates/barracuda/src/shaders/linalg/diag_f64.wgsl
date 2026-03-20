// SPDX-License-Identifier: AGPL-3.0-or-later
// diag_f64.wgsl - Diagonal matrix operations (f64 canonical)
//
// Two modes:
// 1. Extract diagonal from matrix → vector
// 2. Create diagonal matrix from vector → matrix

struct Params {
    size: u32,           // Size of diagonal (N for NxN matrix)
    output_size: u32,     // Elements to process: size for extract, size*size for create
    mode: u32,           // 0 = extract, 1 = create
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.output_size) {
        return;
    }
    
    if (params.mode == 0u) {
        // Extract diagonal: matrix → vector
        // Input: NxN matrix, Output: N-element vector
        let diag_idx = idx * params.size + idx; // [i, i]
        output[idx] = input[diag_idx];
    } else {
        // Create diagonal matrix: vector → matrix
        // Input: N-element vector, Output: NxN matrix (zero except diagonal)
        let row = idx / params.size;
        let col = idx % params.size;
        
        if (row == col) {
            output[idx] = input[row];
        } else {
            output[idx] = f64(0.0);
        }
    }
}
