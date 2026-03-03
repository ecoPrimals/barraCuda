// Matrix Rank - Compute rank via Gaussian elimination (complete implementation) - f64 canonical
// Counts linearly independent rows/columns using row reduction
//
// Algorithm:
// 1. Perform Gaussian elimination with partial pivoting
// 2. Count non-zero rows (rank)
//
// Note: This is a simplified version suitable for small matrices.
// For large matrices, use batched LU decomposition.

struct Params {
    rows: u32,
    cols: u32,
    tolerance: f64,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> matrix: array<f64>;        // [rows, cols]
@group(0) @binding(2) var<storage, read_write> work_matrix: array<f64>; // Working copy
@group(0) @binding(3) var<storage, read_write> rank_buffer: array<u32>; // [1] - output rank

// Step 1: Copy matrix to working buffer
@compute @workgroup_size(256)
fn copy_matrix(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.rows * params.cols;
    if (idx >= total) {
        return;
    }
    work_matrix[idx] = matrix[idx];
}

// Step 2: Gaussian elimination (sequential for correctness)
// Each workgroup processes one pivot
@compute @workgroup_size(1)
fn gaussian_elimination(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pivot_row = global_id.x;
    if (pivot_row >= params.rows || pivot_row >= params.cols) {
        return;
    }

    // Find pivot (max absolute value in column)
    var max_val = abs(work_matrix[pivot_row * params.cols + pivot_row]);
    var max_row = pivot_row;
    
    for (var r = pivot_row + 1u; r < params.rows; r++) {
        let val = abs(work_matrix[r * params.cols + pivot_row]);
        if (val > max_val) {
            max_val = val;
            max_row = r;
        }
    }
    
    // Skip if pivot is zero (within tolerance)
    if (max_val < params.tolerance) {
        return;
    }
    
    // Swap rows if needed
    if (max_row != pivot_row) {
        for (var c = 0u; c < params.cols; c++) {
            let temp = work_matrix[pivot_row * params.cols + c];
            work_matrix[pivot_row * params.cols + c] = work_matrix[max_row * params.cols + c];
            work_matrix[max_row * params.cols + c] = temp;
        }
    }
    
    // Eliminate column below pivot
    let pivot_val = work_matrix[pivot_row * params.cols + pivot_row];
    for (var r = pivot_row + 1u; r < params.rows; r++) {
        let factor = work_matrix[r * params.cols + pivot_row] / pivot_val;
        for (var c = pivot_row; c < params.cols; c++) {
            work_matrix[r * params.cols + c] -= factor * work_matrix[pivot_row * params.cols + c];
        }
    }
}

// Step 3: Count non-zero rows (rank)
@compute @workgroup_size(1)
fn count_rank(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var rank = 0u;
    
    for (var r = 0u; r < params.rows && r < params.cols; r++) {
        // Check if row has any non-zero element
        var has_nonzero = false;
        for (var c = r; c < params.cols; c++) {
            if (abs(work_matrix[r * params.cols + c]) >= params.tolerance) {
                has_nonzero = true;
                break;
            }
        }
        if (has_nonzero) {
            rank += 1u;
        }
    }
    
    rank_buffer[0] = rank;
}
