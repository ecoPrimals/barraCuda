// SPDX-License-Identifier: AGPL-3.0-only
// Triangular Solve (f64) — Forward and Backward Substitution
//
// Solves L·x = b (forward) or U·x = b (backward) for triangular matrices.
//
// **Deep Debt Evolution (Feb 16, 2026)**:
// - ✅ Pure WGSL f64 implementation  
// - ✅ Native f64 arithmetic — Vulkan/WebGPU builtins
// - ✅ Hardware-agnostic (NVIDIA/AMD/Intel via WebGPU)
// - ✅ Science-grade precision (1e-14 tolerance)
// - ✅ WGSL as unified math language
//
// Use Case: After Cholesky decomposition A = L·Lᵀ, solve A·x = b via:
// 1. L·z = b (forward substitution)
// 2. Lᵀ·x = z (backward substitution)
//
// Also used for:
// - LU solve (L·y = b, then U·x = y)
// - QR solve (Q is orthogonal, so R·x = Qᵀ·b)
// - Preconditioning

struct Params {
    n: u32,         // Matrix size (n x n)
    is_lower: u32,  // 1 = lower triangular (forward), 0 = upper (backward)
    is_unit: u32,   // 1 = unit diagonal (don't divide by diag), 0 = general
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f64>;        // Triangular matrix (L or U)
@group(0) @binding(1) var<storage, read> rhs: array<f64>;           // Right-hand side vector b
@group(0) @binding(2) var<storage, read_write> solution: array<f64>; // Solution vector x
@group(0) @binding(3) var<uniform> params: Params;

// Helper for f64 constants (Naga workaround)
fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

@compute @workgroup_size(1)
fn triangular_solve_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    
    if (global_id.x != 0u) {
        return;
    }
    
    // Reference for f64 constant construction
    let ref_val = rhs[0];
    let epsilon = f64_const(ref_val, 1e-14);
    let zero = f64_const(ref_val, 0.0);
    let one = f64_const(ref_val, 1.0);
    
    if (params.is_lower == 1u) {
        // Forward substitution: L·x = b
        // x[i] = (b[i] - Σ L[i,j]*x[j] for j<i) / L[i,i]
        for (var i = 0u; i < n; i = i + 1u) {
            var sum = zero;
            
            for (var j = 0u; j < i; j = j + 1u) {
                sum = sum + matrix[i * n + j] * solution[j];
            }
            
            let diag = matrix[i * n + i];
            
            if (params.is_unit == 1u) {
                // Unit diagonal: L[i,i] = 1 (implicit)
                solution[i] = rhs[i] - sum;
            } else {
                // General: divide by diagonal
                if (abs(diag) < epsilon) {
                    solution[i] = zero;  // Singular
                    return;
                }
                solution[i] = (rhs[i] - sum) / diag;
            }
        }
    } else {
        // Backward substitution: U·x = b
        // x[i] = (b[i] - Σ U[i,j]*x[j] for j>i) / U[i,i]
        for (var i_rev = 0u; i_rev < n; i_rev = i_rev + 1u) {
            let i = n - 1u - i_rev;
            var sum = zero;
            
            for (var j = i + 1u; j < n; j = j + 1u) {
                sum = sum + matrix[i * n + j] * solution[j];
            }
            
            let diag = matrix[i * n + i];
            
            if (params.is_unit == 1u) {
                solution[i] = rhs[i] - sum;
            } else {
                if (abs(diag) < epsilon) {
                    solution[i] = zero;
                    return;
                }
                solution[i] = (rhs[i] - sum) / diag;
            }
        }
    }
}

// Batched variant: solve multiple systems in parallel
// Useful for solving A·X = B where X, B are matrices (column-by-column)
@compute @workgroup_size(1)
fn triangular_solve_f64_batched(
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let n = params.n;
    let batch_idx = wg_id.x;
    
    // Each batch element uses different RHS/solution vectors
    let vec_offset = batch_idx * n;
    
    let ref_val = rhs[vec_offset];
    let epsilon = f64_const(ref_val, 1e-14);
    let zero = f64_const(ref_val, 0.0);
    
    // Matrix is shared across batches (same L or U)
    if (params.is_lower == 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            var sum = zero;
            for (var j = 0u; j < i; j = j + 1u) {
                sum = sum + matrix[i * n + j] * solution[vec_offset + j];
            }
            
            let diag = matrix[i * n + i];
            
            if (params.is_unit == 1u) {
                solution[vec_offset + i] = rhs[vec_offset + i] - sum;
            } else {
                if (abs(diag) < epsilon) {
                    return;
                }
                solution[vec_offset + i] = (rhs[vec_offset + i] - sum) / diag;
            }
        }
    } else {
        for (var i_rev = 0u; i_rev < n; i_rev = i_rev + 1u) {
            let i = n - 1u - i_rev;
            var sum = zero;
            
            for (var j = i + 1u; j < n; j = j + 1u) {
                sum = sum + matrix[i * n + j] * solution[vec_offset + j];
            }
            
            let diag = matrix[i * n + i];
            
            if (params.is_unit == 1u) {
                solution[vec_offset + i] = rhs[vec_offset + i] - sum;
            } else {
                if (abs(diag) < epsilon) {
                    return;
                }
                solution[vec_offset + i] = (rhs[vec_offset + i] - sum) / diag;
            }
        }
    }
}

// Solve Lᵀ·x = b using stored L (transpose in memory access)
// Used for second step of Cholesky solve: L·z = b, then Lᵀ·x = z
@compute @workgroup_size(1)
fn triangular_solve_transpose_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    
    if (global_id.x != 0u) {
        return;
    }
    
    let ref_val = rhs[0];
    let epsilon = f64_const(ref_val, 1e-14);
    let zero = f64_const(ref_val, 0.0);
    
    // Lᵀ is upper triangular, solve backwards
    // Lᵀ[i,j] = L[j,i] (stored column becomes row)
    for (var i_rev = 0u; i_rev < n; i_rev = i_rev + 1u) {
        let i = n - 1u - i_rev;
        var sum = zero;
        
        // Σ Lᵀ[i,j]*x[j] for j>i = Σ L[j,i]*x[j] for j>i
        for (var j = i + 1u; j < n; j = j + 1u) {
            sum = sum + matrix[j * n + i] * solution[j];  // Note: L[j,i] not L[i,j]
        }
        
        let diag = matrix[i * n + i];  // L[i,i] = Lᵀ[i,i]
        
        if (abs(diag) < epsilon) {
            solution[i] = zero;
            return;
        }
        
        solution[i] = (rhs[i] - sum) / diag;
    }
}
