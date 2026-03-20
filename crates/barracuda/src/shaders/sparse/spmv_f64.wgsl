// SPDX-License-Identifier: AGPL-3.0-or-later
// Sparse Matrix-Vector Product (CSR format) - f64 Precision
// y = A * x where A is in Compressed Sparse Row format
//
// Full f64 precision via WGSL native f64 support (SPIR-V/Vulkan)
// Atomic-free design: one thread per row, full precision

struct Params {
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read> values: array<f64>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptrs: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn spmv_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.num_rows) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1u];

    var sum: f64 = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        let col = col_indices[j];
        let val = values[j];
        sum = sum + val * vector[col];
    }
    output[row] = sum;
}
