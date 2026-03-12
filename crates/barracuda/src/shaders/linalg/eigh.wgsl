// SPDX-License-Identifier: AGPL-3.0-only
// Eigenvalue Decomposition (eigh) - Jacobi algorithm for symmetric matrices
// Computes eigenvalues and eigenvectors of A where A = V·D·Vᵀ
//
// **Deep Debt Principles**:
// - ✅ Pure WGSL implementation (GPU-optimized)
// - ✅ Safe Rust (no unsafe blocks)
// - ✅ Hardware-agnostic via WebGPU
// - ✅ Runtime-configured matrix size
// - ✅ Single-thread controller pattern
//
// Algorithm: Jacobi rotation method (iterative)
// 1. Initialize: work = copy of A, eigenvectors = identity
// 2. Each sweep: find max off-diagonal (p,q), apply rotation to zero it
// 3. Repeat until convergence or max_iter
// 4. Eigenvalues = diagonal of work, Eigenvectors = accumulated rotations
//
// Input:  symmetric matrix A [N, N] (row-major flat)
// Output 1: eigenvalues [N]
// Output 2: eigenvectors [N, N] (column-major: column j is eigenvector for eigenvalue j)

struct Params {
    n: u32,         // Matrix size (n x n)
    max_iter: u32,  // Maximum Jacobi sweeps (default 100)
}

@group(0) @binding(0) var<storage, read> input: array<f32>;              // Input symmetric matrix A
@group(0) @binding(1) var<storage, read_write> work: array<f32>;         // Working copy of A (modified in place)
@group(0) @binding(2) var<storage, read_write> eigenvalues: array<f32>; // Output eigenvalues [n]
@group(0) @binding(3) var<storage, read_write> eigenvectors: array<f32>; // Output eigenvectors [n*n] column-major
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = params.n;
    let max_iter = params.max_iter;
    let epsilon = 1e-8;      // Convergence tolerance

    // Single-thread controller (like cholesky.wgsl)
    if (global_id.x != 0u) {
        return;
    }

    // Step 1: Copy input to work, initialize eigenvectors to identity
    for (var i = 0u; i < n * n; i = i + 1u) {
        work[i] = input[i];
    }
    for (var i = 0u; i < n * n; i = i + 1u) {
        eigenvectors[i] = 0.0;
    }
    for (var i = 0u; i < n; i = i + 1u) {
        eigenvectors[i * n + i] = 1.0;
    }

    // Step 2: Jacobi iterations
    for (var iter = 0u; iter < max_iter; iter = iter + 1u) {
        // Find (p,q) with maximum |work[p,q]| for p < q
        var max_off = 0.0;
        var best_p = 0u;
        var best_q = 1u;

        for (var p = 0u; p < n; p = p + 1u) {
            for (var q = p + 1u; q < n; q = q + 1u) {
                let val = abs(work[p * n + q]);
                if (val > max_off) {
                    max_off = val;
                    best_p = p;
                    best_q = q;
                }
            }
        }

        // Check convergence
        if (max_off < epsilon) {
            break;
        }

        let p = best_p;
        let q = best_q;

        // Compute rotation angle: theta = 0.5 * atan2(2*A[p,q], A[q,q]-A[p,p])
        let apq = work[p * n + q];
        let app = work[p * n + p];
        let aqq = work[q * n + q];
        let theta = 0.5 * atan2(2.0 * apq, aqq - app);
        let c = cos(theta);
        let s = sin(theta);

        // Apply rotation to work matrix: A' = J^T · A · J
        // Update rows and columns p, q
        for (var i = 0u; i < n; i = i + 1u) {
            if (i != p && i != q) {
                let a_ip = work[i * n + p];
                let a_iq = work[i * n + q];
                work[i * n + p] = c * a_ip - s * a_iq;
                work[i * n + q] = s * a_ip + c * a_iq;
            }
        }
        for (var i = 0u; i < n; i = i + 1u) {
            if (i != p && i != q) {
                let a_pi = work[p * n + i];
                let a_qi = work[q * n + i];
                work[p * n + i] = c * a_pi - s * a_qi;
                work[q * n + i] = s * a_pi + c * a_qi;
            }
        }

        // Update 2x2 block [p,q] x [p,q]
        work[p * n + p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        work[q * n + q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        work[p * n + q] = 0.0;
        work[q * n + p] = 0.0;

        // Apply rotation to eigenvectors: V' = V · J
        for (var i = 0u; i < n; i = i + 1u) {
            let v_ip = eigenvectors[i * n + p];
            let v_iq = eigenvectors[i * n + q];
            eigenvectors[i * n + p] = c * v_ip - s * v_iq;
            eigenvectors[i * n + q] = s * v_ip + c * v_iq;
        }
    }

    // Step 3: Extract eigenvalues from diagonal
    for (var i = 0u; i < n; i = i + 1u) {
        eigenvalues[i] = work[i * n + i];
    }
}
