// SPDX-License-Identifier: AGPL-3.0-or-later
// Sparse matrix multiply with quantized int8 values
// Uses COO (coordinate) format for sparse matrix

struct Params {
    nnz: u32,
    output_size: u32,
    scale: f32,
    padding: u32,
}

@group(0) @binding(0) var<storage, read> values: array<i32>;     // Sparse values (int8 as i32)
@group(0) @binding(1) var<storage, read> rows: array<u32>;       // Row indices
@group(0) @binding(2) var<storage, read> cols: array<u32>;       // Column indices
@group(0) @binding(3) var<storage, read> dense: array<i32>;      // Dense vector (int8 as i32)
@group(0) @binding(4) var<storage, read_write> output: array<f32>; // Output (dequantized)
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn sparse_matmul_quantized(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row_idx = gid.x;
    if (row_idx >= params.output_size) {
        return;
    }
    
    // Accumulate all non-zero entries for this row
    var sum = 0i;
    for (var i = 0u; i < params.nnz; i = i + 1u) {
        if (rows[i] == row_idx) {
            let col = cols[i];
            let val = values[i];
            let dense_val = dense[col];
            sum = sum + (val * dense_val);
        }
    }
    
    // Dequantize: convert int32 accumulator to fp32 and scale
    output[row_idx] = f32(sum) * params.scale;
}
