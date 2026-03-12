// SPDX-License-Identifier: AGPL-3.0-only
// Cholesky Decomposition (f64) — L·Lᵀ factorization for scientific computing
//
// Computes lower triangular matrix L such that A = L·Lᵀ
// for symmetric positive-definite (SPD) matrices.
//
// **Deep Debt Evolution (Feb 16, 2026)**:
// - ✅ Pure WGSL f64 implementation
// - ✅ Native sqrt(f64) — Vulkan/WebGPU supports f64 builtins
// - ✅ Hardware-agnostic (NVIDIA/AMD/Intel via WebGPU)
// - ✅ Science-grade precision (1e-14 tolerance)
// - ✅ WGSL as unified math language (same code, any GPU)
//
// Algorithm: Standard Cholesky (column-by-column)
// 1. For each column j:
//    a. L[j,j] = sqrt(A[j,j] - Σ L[j,k]² for k<j)
//    b. For i > j: L[i,j] = (A[i,j] - Σ L[i,k]*L[j,k]) / L[j,j]
//
// Use cases:
// - Solving SPD systems (covariance matrices, Gram matrices)
// - Gaussian process regression (Krylov preconditioner)
// - Optimization (Hessian factorization)
// - Nuclear physics (overlap matrices)

struct Params {
    n: u32,  // Matrix size (n x n)
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // Input SPD matrix A
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // Output lower triangular L
@group(0) @binding(2) var<uniform> params: Params;

// Helper to get f64 constant (Naga/WGSL gotcha workaround)
fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

@compute @workgroup_size(256)
fn cholesky_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    
    // Single-threaded column-by-column decomposition
    // (Future evolution: blocked parallel Cholesky for large n)
    if (global_id.x != 0u) {
        return;
    }
    
    // f64 epsilon for numerical stability (smaller than f32's 1e-8)
    // Using a reference value to satisfy Naga's AbstractFloat rules
    let ref_val = input[0];
    let epsilon = f64_const(ref_val, 1e-14);
    let zero = f64_const(ref_val, 0.0);
    
    // Initialize output to zero
    for (var i = 0u; i < n * n; i = i + 1u) {
        output[i] = zero;
    }
    
    // Cholesky decomposition: column-by-column
    for (var j = 0u; j < n; j = j + 1u) {
        // Compute L[j,j] = sqrt(A[j,j] - Σ L[j,k]² for k<j)
        var sum_sq = zero;
        for (var k = 0u; k < j; k = k + 1u) {
            let l_jk = output[j * n + k];
            sum_sq = sum_sq + l_jk * l_jk;
        }
        
        let diag_val = input[j * n + j] - sum_sq;
        
        // Check for positive definiteness
        if (diag_val <= epsilon) {
            // Matrix is not positive definite — leave zeros as error
            return;
        }
        
        // Native f64 sqrt — full hardware acceleration via Vulkan
        output[j * n + j] = sqrt(diag_val);
        let l_jj = output[j * n + j];
        
        // Compute L[i,j] for i > j (below diagonal)
        for (var i = j + 1u; i < n; i = i + 1u) {
            var sum_prod = zero;
            for (var k = 0u; k < j; k = k + 1u) {
                sum_prod = sum_prod + output[i * n + k] * output[j * n + k];
            }
            
            output[i * n + j] = (input[i * n + j] - sum_prod) / l_jj;
        }
    }
}

// Batched variant: one matrix per workgroup
// For diagonalizing covariance matrices in parallel
@compute @workgroup_size(1)
fn cholesky_f64_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let n = params.n;
    let batch_idx = wg_id.x;
    let mat_size = n * n;
    let in_offset = batch_idx * mat_size;
    let out_offset = batch_idx * mat_size;
    
    // Each workgroup processes one matrix
    if (gid.x != batch_idx) {
        return;
    }
    
    let ref_val = input[in_offset];
    let epsilon = f64_const(ref_val, 1e-14);
    let zero = f64_const(ref_val, 0.0);
    
    // Initialize output to zero
    for (var i = 0u; i < mat_size; i = i + 1u) {
        output[out_offset + i] = zero;
    }
    
    // Cholesky decomposition
    for (var j = 0u; j < n; j = j + 1u) {
        var sum_sq = zero;
        for (var k = 0u; k < j; k = k + 1u) {
            let l_jk = output[out_offset + j * n + k];
            sum_sq = sum_sq + l_jk * l_jk;
        }
        
        let diag_val = input[in_offset + j * n + j] - sum_sq;
        
        if (diag_val <= epsilon) {
            return;  // Not SPD
        }
        
        output[out_offset + j * n + j] = sqrt(diag_val);
        let l_jj = output[out_offset + j * n + j];
        
        for (var i = j + 1u; i < n; i = i + 1u) {
            var sum_prod = zero;
            for (var k = 0u; k < j; k = k + 1u) {
                sum_prod = sum_prod + output[out_offset + i * n + k] * output[out_offset + j * n + k];
            }
            
            output[out_offset + i * n + j] = (input[in_offset + i * n + j] - sum_prod) / l_jj;
        }
    }
}
