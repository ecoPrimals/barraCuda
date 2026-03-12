// SPDX-License-Identifier: AGPL-3.0-only
// Linear System Solve - Gaussian elimination with partial pivoting
// Solves A·x = b for square matrix A and vector b
//
// **Deep Debt Principles**:
// - ✅ Pure WGSL implementation (GPU-optimized)
// - ✅ Safe Rust (no unsafe blocks)
// - ✅ Hardware-agnostic via WebGPU
// - ✅ Runtime-configured matrix size
// - ✅ Single-thread controller pattern
//
// Algorithm: Gaussian elimination with row pivoting
// 1. Copy A to work[0..n²], b to work[n²..n²+n]
// 2. For each column k: find pivot row, swap, eliminate
// 3. Result: work[0..n²] is upper triangular U, work[n²..] is modified rhs y
// 4. Back substitution: overwrite work[n²..] with solution x
//
// Input 1: matrix A [N, N]
// Input 2: vector b [N]
// Output: solution x [N] where A·x = b (stored in work buffer last n elements)

struct Params {
    n: u32,  // Matrix size (n x n)
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;       // Input matrix A
@group(0) @binding(1) var<storage, read> rhs: array<f32>;         // Input vector b
@group(0) @binding(2) var<storage, read_write> output: array<f32>; // Work + solution: [0..n²]=U, [n²..n²+n]=x
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    let nn = n * n;
    let epsilon = 1e-10;

    // Single-thread controller (like cholesky.wgsl)
    if (global_id.x != 0u) {
        return;
    }

    // Step 1: Copy matrix to output[0..n²], rhs to output[n²..n²+n]
    for (var i = 0u; i < nn; i = i + 1u) {
        output[i] = matrix[i];
    }
    for (var i = 0u; i < n; i = i + 1u) {
        output[nn + i] = rhs[i];
    }

    // Step 2: Gaussian elimination with partial pivoting
    for (var k = 0u; k < n; k = k + 1u) {
        // Find pivot: max |output[i*n+k]| for i >= k
        var max_val = 0.0;
        var pivot_row = k;
        for (var i = k; i < n; i = i + 1u) {
            let val = abs(output[i * n + k]);
            if (val > max_val) {
                max_val = val;
                pivot_row = i;
            }
        }

        // Swap row k with pivot_row in matrix part
        if (pivot_row != k) {
            for (var j = 0u; j < n; j = j + 1u) {
                let tmp = output[k * n + j];
                output[k * n + j] = output[pivot_row * n + j];
                output[pivot_row * n + j] = tmp;
            }
            let tmp_b = output[nn + k];
            output[nn + k] = output[nn + pivot_row];
            output[nn + pivot_row] = tmp_b;
        }

        // Check for singularity
        let pivot = output[k * n + k];
        if (abs(pivot) < epsilon) {
            output[nn] = 0.0;  // Error indicator
            return;
        }

        // Eliminate column k below pivot
        for (var i = k + 1u; i < n; i = i + 1u) {
            let factor = output[i * n + k] / pivot;
            output[i * n + k] = 0.0;
            for (var j = k + 1u; j < n; j = j + 1u) {
                output[i * n + j] = output[i * n + j] - factor * output[k * n + j];
            }
            output[nn + i] = output[nn + i] - factor * output[nn + k];
        }
    }

    // Step 3: Back substitution (output[0..n²] is upper triangular)
    // Overwrite output[n²..n²+n] with solution x
    for (var i_rev = 0u; i_rev < n; i_rev = i_rev + 1u) {
        let i = n - 1u - i_rev;
        var sum = 0.0;
        for (var j = i + 1u; j < n; j = j + 1u) {
            sum = sum + output[i * n + j] * output[nn + j];
        }
        let diag = output[i * n + i];
        if (abs(diag) < epsilon) {
            output[nn + i] = 0.0;
            return;
        }
        output[nn + i] = (output[nn + i] - sum) / diag;
    }
}
