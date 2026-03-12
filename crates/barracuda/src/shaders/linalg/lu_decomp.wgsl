// SPDX-License-Identifier: AGPL-3.0-only
// LU Decomposition - Shader-First Implementation
//
// Parallel row elimination for LU factorization: A = LU
//
// The algorithm has two parts:
// 1. PIVOT SELECTION (sequential on CPU) - find max in column
// 2. ROW ELIMINATION (parallel on GPU) - update all rows simultaneously
//
// For each column k:
//   - CPU selects pivot (sequential scan of column)
//   - GPU updates all rows i > k in parallel: A[i,j] -= (A[i,k]/A[k,k]) * A[k,j]
//
// This is SHADER-FIRST LU:
// - Pivot selection remains on CPU (O(n) per column)
// - Row updates are O(n²) parallel operations per column
// - Total: O(n) sequential columns × O(n²) parallel work = O(n³) with O(n²) parallelism
//
// For future quantum/fp128 hardware:
// - Same shaders run on any compute substrate
// - Pivot selection could be parallelized with different hardware
//
// Reference: Golub & Van Loan, "Matrix Computations" (2013)

struct LuParams {
    n: u32,           // Matrix dimension
    k: u32,           // Current elimination column
    pivot_row: u32,   // Row with pivot (from CPU selection)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: LuParams;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;    // Matrix [n × n], row-major
@group(0) @binding(2) var<storage, read_write> perm: array<u32>; // Permutation vector [n]

// Row swap: Swap rows pivot_row and k
// Parallel over columns
@compute @workgroup_size(256, 1, 1)
fn row_swap(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;  // Column index
    let n = params.n;
    let k = params.k;
    let pivot = params.pivot_row;
    
    if (j >= n || k == pivot) {
        return;
    }
    
    let idx_k = k * n + j;
    let idx_pivot = pivot * n + j;
    
    let temp = A[idx_k];
    A[idx_k] = A[idx_pivot];
    A[idx_pivot] = temp;
}

// Compute L multipliers: L[i,k] = A[i,k] / A[k,k]
// Parallel over rows i > k
@compute @workgroup_size(256, 1, 1)
fn compute_multipliers(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x + params.k + 1u;  // Row index (i > k)
    let n = params.n;
    let k = params.k;
    
    if (i >= n) {
        return;
    }
    
    let pivot_val = A[k * n + k];
    if (abs(pivot_val) > 1e-10) {
        // Store L[i,k] in place of A[i,k] (below diagonal)
        A[i * n + k] = A[i * n + k] / pivot_val;
    }
}

// Row elimination: A[i,j] -= L[i,k] * A[k,j] for all i > k, j > k
// Parallel over all (i,j) pairs
@compute @workgroup_size(16, 16, 1)
fn row_elimination(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let local_i = global_id.y;  // Relative row index
    let local_j = global_id.x;  // Relative column index
    let n = params.n;
    let k = params.k;
    
    let i = local_i + k + 1u;  // Actual row (i > k)
    let j = local_j + k + 1u;  // Actual column (j > k)
    
    if (i >= n || j >= n) {
        return;
    }
    
    let L_ik = A[i * n + k];    // Multiplier (already computed)
    let A_kj = A[k * n + j];    // Row k element
    
    A[i * n + j] = A[i * n + j] - L_ik * A_kj;
}

// Forward substitution: Solve Ly = b (L is unit lower triangular)
// Each row depends on previous rows, so we do one row at a time
// But within a row, the dot product is parallel
struct SolveParams {
    n: u32,
    row: u32,         // Current row being solved
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> solve_params: SolveParams;
@group(0) @binding(1) var<storage, read> L: array<f32>;      // Lower triangular [n × n]
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // Solution/RHS [n]

var<workgroup> partial_dot: array<f32, 256>;

// Compute y[row] = b[row] - Σ_{j<row} L[row,j] * y[j]
// The sum is a parallel reduction
@compute @workgroup_size(256, 1, 1)
fn forward_sub_row(@builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = local_id.x;
    let row = solve_params.row;
    let n = solve_params.n;
    
    // Each thread handles multiple elements of the sum
    var sum: f32 = 0.0;
    var j = tid;
    while (j < row) {
        sum = sum + L[row * n + j] * y[j];
        j = j + 256u;
    }
    
    partial_dot[tid] = sum;
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            partial_dot[tid] = partial_dot[tid] + partial_dot[tid + stride];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        y[row] = y[row] - partial_dot[0];
    }
}

// Back substitution: Solve Ux = y (U is upper triangular)
// Similar pattern: one row at a time, parallel dot product within row
@group(0) @binding(0) var<uniform> back_params: SolveParams;
@group(0) @binding(1) var<storage, read> U: array<f32>;      // Upper triangular [n × n]
@group(0) @binding(2) var<storage, read_write> x: array<f32>; // Solution [n]

var<workgroup> partial_back: array<f32, 256>;

// Compute x[row] = (y[row] - Σ_{j>row} U[row,j] * x[j]) / U[row,row]
@compute @workgroup_size(256, 1, 1)
fn back_sub_row(@builtin(local_invocation_id) local_id: vec3<u32>,
                @builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = local_id.x;
    let row = back_params.row;
    let n = back_params.n;
    
    // Sum over j > row
    var sum: f32 = 0.0;
    var j = row + 1u + tid;
    while (j < n) {
        sum = sum + U[row * n + j] * x[j];
        j = j + 256u;
    }
    
    partial_back[tid] = sum;
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            partial_back[tid] = partial_back[tid] + partial_back[tid + stride];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        let diag = U[row * n + row];
        if (abs(diag) > 1e-10) {
            x[row] = (x[row] - partial_back[0]) / diag;
        }
    }
}

// Find pivot: Find max absolute value in column k (rows k to n-1)
// Returns the row index - used for CPU orchestration
// This is a parallel reduction returning an index
struct PivotParams {
    n: u32,
    k: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> pivot_params: PivotParams;
@group(0) @binding(1) var<storage, read> A_pivot: array<f32>;
@group(0) @binding(2) var<storage, read_write> pivot_result: array<u32>;  // [row_idx, max_val_bits]

var<workgroup> shared_vals: array<f32, 256>;
var<workgroup> shared_idxs: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn find_pivot(@builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = local_id.x;
    let n = pivot_params.n;
    let k = pivot_params.k;
    
    // Each thread finds max in its portion
    var max_val: f32 = 0.0;
    var max_idx: u32 = k;
    
    var i = k + tid;
    while (i < n) {
        let val = abs(A_pivot[i * n + k]);
        if (val > max_val) {
            max_val = val;
            max_idx = i;
        }
        i = i + 256u;
    }
    
    shared_vals[tid] = max_val;
    shared_idxs[tid] = max_idx;
    workgroupBarrier();
    
    // Parallel reduction to find global max
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            if (shared_vals[tid + stride] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + stride];
                shared_idxs[tid] = shared_idxs[tid + stride];
            }
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        pivot_result[0] = shared_idxs[0];
        pivot_result[1] = bitcast<u32>(shared_vals[0]);
    }
}
