// SPDX-License-Identifier: AGPL-3.0-or-later
// Matrix Multiplication (fp64 - High Precision)
//
// **Purpose**: High-precision matrix multiplication for scientific computing
// **Precision**: f64 (IEEE 754 double precision, ~15 decimal digits)
// **Use Cases**: Numerical analysis, gradient accumulation, physics simulation
//
// **vs fp32**:
// - fp64: ~15 digits precision, range ±10^308
// - fp32: ~7 digits precision, range ±10^38
// - Performance: fp64 is typically 2-32x slower on consumer GPUs
//
// **When to use fp64**:
// ✅ Scientific computing requiring high precision
// ✅ Long-running iterative algorithms (gradient accumulation)
// ✅ Numerical stability in ill-conditioned problems
// ❌ Real-time inference (use fp32)
// ❌ FHE operations (use exact integer arithmetic)

// Input matrices
@group(0) @binding(0) var<storage, read> matrix_a: array<f64>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f64>;

// Matrix dimensions
struct MatmulParams {
    m: u32,  // Rows of A
    n: u32,  // Cols of A, rows of B
    p: u32,  // Cols of B
}

@group(0) @binding(3) var<uniform> params: MatmulParams;

/// High-precision matrix multiplication kernel
///
/// Computes: C = A × B where
/// - A is m × n
/// - B is n × p  
/// - C is m × p
///
/// Each thread computes one element C[i,j] = Σ A[i,k] * B[k,j]
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    
    // Bounds check
    if (row >= params.m || col >= params.p) {
        return;
    }
    
    // Compute dot product using Kahan summation for precision
    var sum = 0.0;
    var compensation = 0.0;  // Kahan summation error correction
    
    for (var k = 0u; k < params.n; k = k + 1u) {
        let a_val = matrix_a[row * params.n + k];
        let b_val = matrix_b[k * params.p + col];
        
        // Kahan summation for numerical stability
        let y = a_val * b_val - compensation;
        let t = sum + y;
        compensation = (t - sum) - y;
        sum = t;
    }
    
    matrix_c[row * params.p + col] = sum;
}

// ═══════════════════════════════════════════════════════════════
// Performance Notes
// ═══════════════════════════════════════════════════════════════
//
// Expected Performance (fp64 vs fp32):
//   Consumer GPUs:  2-32x slower
//   Workstation:    2-8x slower
//   Server GPUs:    1-2x slower (Tesla V100, A100)
//
// When fp64 is worth it:
//   - Scientific computing (molecular dynamics, climate modeling)
//   - Financial modeling (risk calculations)
//   - Deep learning research (gradient precision)
//   - Numerical solvers (ill-conditioned systems)
//
// Precision comparison (summing 1e8 numbers):
//   fp32: Error ~1e-2 (unacceptable for science)
//   fp64: Error ~1e-10 (excellent)
//   fp64 + Kahan: Error ~1e-15 (near-perfect)
