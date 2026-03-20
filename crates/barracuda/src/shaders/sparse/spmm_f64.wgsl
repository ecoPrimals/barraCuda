// SPDX-License-Identifier: AGPL-3.0-or-later
// Sparse Matrix-Matrix Product (CSR × Dense, f64)
//
// C[M, N] = A[M, K] × B[K, N]
// where A is CSR (values, col_indices, row_ptr) and B is dense row-major.
//
// One thread per (row, output_col) pair. Each thread iterates over non-zeros
// in the CSR row and accumulates A[row, k] * B[k, col].

struct SpmmParams {
    m: u32,       // rows in A / rows in C
    k: u32,       // cols in A / rows in B (unused in CSR traversal but kept for validation)
    n: u32,       // cols in B / cols in C
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       a_values:      array<f64>;
@group(0) @binding(1) var<storage, read>       a_col_indices: array<u32>;
@group(0) @binding(2) var<storage, read>       a_row_ptr:     array<u32>;
@group(0) @binding(3) var<storage, read>       b:             array<f64>;
@group(0) @binding(4) var<storage, read_write> c:             array<f64>;
@group(0) @binding(5) var<uniform>             params:        SpmmParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.m * params.n;
    if (flat >= total) { return; }

    let row = flat / params.n;
    let col = flat % params.n;

    let start = a_row_ptr[row];
    let end   = a_row_ptr[row + 1u];

    var sum: f64 = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        let k_idx = a_col_indices[j];
        let a_val = a_values[j];
        let b_val = b[k_idx * params.n + col];
        sum = sum + a_val * b_val;
    }

    c[row * params.n + col] = sum;
}
