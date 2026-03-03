// Triangular Solve - Forward and Backward Substitution
// Solves L·x = b (forward) or Uᵀ·x = b (backward) for triangular matrices
//
// **Deep Debt Principles**:
// - ✅ Pure WGSL implementation (GPU-optimized)
// - ✅ Safe Rust (no unsafe blocks)
// - ✅ Hardware-agnostic via WebGPU
// - ✅ Runtime-configured matrix size
//
// Use Case: After Cholesky decomposition A = L·Lᵀ, solve:
// 1. L·z = b (forward substitution)
// 2. Lᵀ·x = z (backward substitution)
// Result: x is the solution to A·x = b
//
// Algorithm: Sequential substitution with dependency chain
// Forward:  x[i] = (b[i] - Σ L[i,j]*x[j] for j<i) / L[i,i]
// Backward: x[i] = (b[i] - Σ U[i,j]*x[j] for j>i) / U[i,i]

struct Params {
    n: u32,         // Matrix size (n x n)
    is_lower: u32,  // 1 for lower triangular (forward), 0 for upper (backward)
}

@group(0) @binding(0) var<storage, read> matrix: array<f32>;   // Triangular matrix (L or U)
@group(0) @binding(1) var<storage, read> rhs: array<f32>;      // Right-hand side vector b
@group(0) @binding(2) var<storage, read_write> solution: array<f32>;  // Solution vector x
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    let epsilon = 1e-10;
    
    // Check if this is the controller thread
    if (global_id.x != 0u) {
        return;
    }
    
    if (params.is_lower == 1u) {
        // Forward substitution: L·x = b
        // Solve for x starting from top (x[0], x[1], ...)
        for (var i = 0u; i < n; i = i + 1u) {
            var sum = 0.0;
            
            // Subtract known terms: Σ L[i,j]*x[j] for j<i
            for (var j = 0u; j < i; j = j + 1u) {
                sum = sum + matrix[i * n + j] * solution[j];
            }
            
            let diag = matrix[i * n + i];
            
            // Check for zero diagonal (singular matrix)
            if (abs(diag) < epsilon) {
                solution[i] = 0.0;  // Error indicator
                return;
            }
            
            // x[i] = (b[i] - sum) / L[i,i]
            solution[i] = (rhs[i] - sum) / diag;
        }
    } else {
        // Backward substitution: Uᵀ·x = b (or U·x = b)
        // Solve for x starting from bottom (x[n-1], x[n-2], ...)
        for (var i_rev = 0u; i_rev < n; i_rev = i_rev + 1u) {
            let i = n - 1u - i_rev;
            var sum = 0.0;
            
            // Subtract known terms: Σ U[i,j]*x[j] for j>i
            for (var j = i + 1u; j < n; j = j + 1u) {
                sum = sum + matrix[i * n + j] * solution[j];
            }
            
            let diag = matrix[i * n + i];
            
            // Check for zero diagonal (singular matrix)
            if (abs(diag) < epsilon) {
                solution[i] = 0.0;  // Error indicator
                return;
            }
            
            // x[i] = (b[i] - sum) / U[i,i]
            solution[i] = (rhs[i] - sum) / diag;
        }
    }
}
