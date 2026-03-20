// SPDX-License-Identifier: AGPL-3.0-or-later
// Batched Symmetric Eigenvalue Decomposition (f64) - Shader-First Implementation
//
// Processes multiple symmetric matrices simultaneously.
// Use case: HFB Hamiltonian diagonalization (52 nuclei, 20-50 dim each)
//
// Each workgroup handles one matrix from the batch.
// Full f64 precision via WGSL native f64 support (SPIR-V/Vulkan)
//
// Memory layout:
//   A_batch: [batch_size × n × n] f64 - Input symmetric matrices, row-major
//   V_batch: [batch_size × n × n] f64 - Output eigenvectors, row-major
//   eigenvalues_batch: [batch_size × n] f64 - Output eigenvalues
//
// Algorithm: Jacobi eigenvalue (same as single-matrix version)
// Each batch element runs independently.

struct BatchedEighParams {
    n: u32,           // Matrix dimension (same for all matrices in batch)
    batch_size: u32,  // Number of matrices
    max_sweeps: u32,  // Maximum Jacobi sweeps
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: BatchedEighParams;
@group(0) @binding(1) var<storage, read_write> A_batch: array<f64>;
@group(0) @binding(2) var<storage, read_write> V_batch: array<f64>;
@group(0) @binding(3) var<storage, read_write> eigenvalues_batch: array<f64>;

// Helper: Get matrix offset for batch index
fn matrix_offset(batch_idx: u32, n: u32) -> u32 {
    return batch_idx * n * n;
}

// Helper: Get eigenvalue offset for batch index
fn eigenvalue_offset(batch_idx: u32, n: u32) -> u32 {
    return batch_idx * n;
}

// Initialize V = Identity for all matrices in batch
// Dispatch with (n, n, batch_size) workgroups
@compute @workgroup_size(16, 16, 1)
fn batched_init_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let col = global_id.x;
    let row = global_id.y;
    let batch_idx = global_id.z;
    let n = params.n;

    if (row >= n || col >= n || batch_idx >= params.batch_size) {
        return;
    }

    let base = matrix_offset(batch_idx, n);
    let idx = base + row * n + col;

    // Naga resolves 1.0 as f32; use f64() constructor for storage writes
    if (row == col) {
        V_batch[idx] = f64(1.0);
    } else {
        V_batch[idx] = f64(0.0);
    }
}

// Extract eigenvalues from diagonal of A for all matrices
// Dispatch with (n, batch_size, 1)
@compute @workgroup_size(256, 1, 1)
fn batched_extract_eigenvalues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let batch_idx = global_id.y;
    let n = params.n;

    if (i >= n || batch_idx >= params.batch_size) {
        return;
    }

    let a_base = matrix_offset(batch_idx, n);
    let e_base = eigenvalue_offset(batch_idx, n);

    eigenvalues_batch[e_base + i] = A_batch[a_base + i * n + i];
}

// ============================================================================
// Single-matrix Jacobi kernels with batch index parameter
// These are dispatched per-matrix with batch_idx specified in params
// ============================================================================

struct BatchedRotParams {
    n: u32,
    batch_idx: u32,   // Which matrix in the batch
    p: u32,           // First pivot index
    q: u32,           // Second pivot index (q > p)
}

@group(0) @binding(0) var<uniform> rot_params: BatchedRotParams;
@group(0) @binding(1) var<storage, read_write> A_rot: array<f64>;
@group(0) @binding(2) var<storage, read_write> cs_batch: array<f64>;  // [batch_size × 2]

// Compute Jacobi rotation angle for one matrix
// Outputs c = cos(θ), s = sin(θ) to cs_batch[batch_idx * 2]
@compute @workgroup_size(1, 1, 1)
fn batched_compute_jacobi_angle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = rot_params.n;
    let batch_idx = rot_params.batch_idx;
    let p = rot_params.p;
    let q = rot_params.q;

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let app = A_rot[base + p * n + p];
    let aqq = A_rot[base + q * n + q];
    let apq = A_rot[base + p * n + q];

    // Handle near-zero off-diagonal
    if (abs(apq) < 1e-14) {
        cs_batch[cs_base] = f64(1.0);       // c = 1
        cs_batch[cs_base + 1u] = f64(0.0);  // s = 0
        return;
    }

    let diff = aqq - app;
    var t: f64;

    if (abs(diff) < 1e-14) {
        // app ≈ aqq: use t = sign(apq)
        if (apq >= 0.0) {
            t = f64(1.0);
        } else {
            t = f64(-1.0);
        }
    } else {
        // Standard computation: tan(2θ) = 2*apq / (aqq - app)
        let phi = diff / (2.0 * apq);
        let abs_phi = abs(phi);
        if (phi >= 0.0) {
            t = f64(1.0) / (abs_phi + sqrt(f64(1.0) + phi * phi));
        } else {
            t = f64(-1.0) / (abs_phi + sqrt(f64(1.0) + phi * phi));
        }
    }

    // c = 1 / sqrt(1 + t²), s = t * c
    let c = f64(1.0) / sqrt(f64(1.0) + t * t);
    let s = t * c;

    cs_batch[cs_base] = c;
    cs_batch[cs_base + 1u] = s;
}

// Apply Jacobi rotation to matrix A (rows and columns p, q)
@group(0) @binding(0) var<uniform> apply_params: BatchedRotParams;
@group(0) @binding(1) var<storage, read_write> A_apply: array<f64>;
@group(0) @binding(2) var<storage, read> cs_apply: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn batched_jacobi_rotate_A(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let n = apply_params.n;
    let batch_idx = apply_params.batch_idx;
    let p = apply_params.p;
    let q = apply_params.q;

    if (k >= n || k == p || k == q) {
        return;
    }

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let c = cs_apply[cs_base];
    let s = cs_apply[cs_base + 1u];

    // Update row k (columns p and q)
    let akp = A_apply[base + k * n + p];
    let akq = A_apply[base + k * n + q];

    let new_akp = c * akp - s * akq;
    let new_akq = s * akp + c * akq;

    A_apply[base + k * n + p] = new_akp;
    A_apply[base + k * n + q] = new_akq;

    // Symmetric: A[p,k] = A[k,p], A[q,k] = A[k,q]
    A_apply[base + p * n + k] = new_akp;
    A_apply[base + q * n + k] = new_akq;
}

// Update the 2×2 block at (p,p), (p,q), (q,p), (q,q)
@compute @workgroup_size(1, 1, 1)
fn batched_jacobi_update_block(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = apply_params.n;
    let batch_idx = apply_params.batch_idx;
    let p = apply_params.p;
    let q = apply_params.q;

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let c = cs_apply[cs_base];
    let s = cs_apply[cs_base + 1u];

    let app = A_apply[base + p * n + p];
    let aqq = A_apply[base + q * n + q];
    let apq = A_apply[base + p * n + q];

    // New diagonal elements after rotation
    let app_new = c * c * app - 2.0 * c * s * apq + s * s * aqq;
    let aqq_new = s * s * app + 2.0 * c * s * apq + c * c * aqq;

    A_apply[base + p * n + p] = app_new;
    A_apply[base + q * n + q] = aqq_new;

    // Off-diagonal should be zero after rotation
    A_apply[base + p * n + q] = f64(0.0);
    A_apply[base + q * n + p] = f64(0.0);
}

// Apply Jacobi rotation to eigenvector matrix V (columns p and q)
@group(0) @binding(0) var<uniform> v_params: BatchedRotParams;
@group(0) @binding(1) var<storage, read_write> V_rot: array<f64>;
@group(0) @binding(2) var<storage, read> cs_v: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn batched_jacobi_rotate_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let n = v_params.n;
    let batch_idx = v_params.batch_idx;
    let p = v_params.p;
    let q = v_params.q;

    if (k >= n) {
        return;
    }

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let c = cs_v[cs_base];
    let s = cs_v[cs_base + 1u];

    // Update row k: V[k,p] and V[k,q]
    let vkp = V_rot[base + k * n + p];
    let vkq = V_rot[base + k * n + q];

    V_rot[base + k * n + p] = c * vkp - s * vkq;
    V_rot[base + k * n + q] = s * vkp + c * vkq;
}

// ============================================================================
// Parallel Jacobi sweep - processes all matrices simultaneously
// Each thread handles one (batch, p, q) tuple
// ============================================================================

// For truly parallel batched execution, we process all matrices and all
// rotation pairs in a single dispatch. This kernel computes the rotation
// angle for one (batch_idx, p, q) pair.
struct ParallelSweepParams {
    n: u32,
    batch_size: u32,
    current_p: u32,  // Current p for cyclic sweep
    current_q: u32,  // Current q for cyclic sweep
}

@group(0) @binding(0) var<uniform> sweep_params: ParallelSweepParams;
@group(0) @binding(1) var<storage, read_write> A_sweep: array<f64>;
@group(0) @binding(2) var<storage, read_write> V_sweep: array<f64>;
@group(0) @binding(3) var<storage, read_write> cs_sweep: array<f64>;  // [batch_size × 2]

// Parallel compute all rotation angles for current (p,q) across all batches
@compute @workgroup_size(64, 1, 1)
fn parallel_compute_angles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= sweep_params.batch_size) {
        return;
    }

    let n = sweep_params.n;
    let p = sweep_params.current_p;
    let q = sweep_params.current_q;

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let app = A_sweep[base + p * n + p];
    let aqq = A_sweep[base + q * n + q];
    let apq = A_sweep[base + p * n + q];

    // Handle near-zero off-diagonal
    if (abs(apq) < 1e-14) {
        cs_sweep[cs_base] = f64(1.0);
        cs_sweep[cs_base + 1u] = f64(0.0);
        return;
    }

    let diff = aqq - app;
    var t: f64;

    if (abs(diff) < 1e-14) {
        if (apq >= 0.0) { t = f64(1.0); } else { t = f64(-1.0); }
    } else {
        let phi = diff / (2.0 * apq);
        let abs_phi = abs(phi);
        if (phi >= 0.0) {
            t = f64(1.0) / (abs_phi + sqrt(f64(1.0) + phi * phi));
        } else {
            t = f64(-1.0) / (abs_phi + sqrt(f64(1.0) + phi * phi));
        }
    }

    let c = f64(1.0) / sqrt(f64(1.0) + t * t);
    let s = t * c;

    cs_sweep[cs_base] = c;
    cs_sweep[cs_base + 1u] = s;
}

// Parallel apply rotation to A for all batches (one row per thread)
// Dispatch: (n, batch_size, 1)
@compute @workgroup_size(64, 1, 1)
fn parallel_rotate_A(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;

    let n = sweep_params.n;
    if (k >= n || batch_idx >= sweep_params.batch_size) {
        return;
    }

    let p = sweep_params.current_p;
    let q = sweep_params.current_q;

    if (k == p || k == q) {
        return;
    }

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let c = cs_sweep[cs_base];
    let s = cs_sweep[cs_base + 1u];

    let akp = A_sweep[base + k * n + p];
    let akq = A_sweep[base + k * n + q];

    let new_akp = c * akp - s * akq;
    let new_akq = s * akp + c * akq;

    A_sweep[base + k * n + p] = new_akp;
    A_sweep[base + k * n + q] = new_akq;
    A_sweep[base + p * n + k] = new_akp;
    A_sweep[base + q * n + k] = new_akq;
}

// Parallel update 2×2 blocks for all batches
@compute @workgroup_size(64, 1, 1)
fn parallel_update_blocks(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    if (batch_idx >= sweep_params.batch_size) {
        return;
    }

    let n = sweep_params.n;
    let p = sweep_params.current_p;
    let q = sweep_params.current_q;

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let c = cs_sweep[cs_base];
    let s = cs_sweep[cs_base + 1u];

    let app = A_sweep[base + p * n + p];
    let aqq = A_sweep[base + q * n + q];
    let apq = A_sweep[base + p * n + q];

    let app_new = c * c * app - 2.0 * c * s * apq + s * s * aqq;
    let aqq_new = s * s * app + 2.0 * c * s * apq + c * c * aqq;

    A_sweep[base + p * n + p] = app_new;
    A_sweep[base + q * n + q] = aqq_new;
    A_sweep[base + p * n + q] = f64(0.0);
    A_sweep[base + q * n + p] = f64(0.0);
}

// Parallel rotate V for all batches
@compute @workgroup_size(64, 1, 1)
fn parallel_rotate_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let batch_idx = global_id.y;

    let n = sweep_params.n;
    if (k >= n || batch_idx >= sweep_params.batch_size) {
        return;
    }

    let p = sweep_params.current_p;
    let q = sweep_params.current_q;

    let base = matrix_offset(batch_idx, n);
    let cs_base = batch_idx * 2u;

    let c = cs_sweep[cs_base];
    let s = cs_sweep[cs_base + 1u];

    let vkp = V_sweep[base + k * n + p];
    let vkq = V_sweep[base + k * n + q];

    V_sweep[base + k * n + p] = c * vkp - s * vkq;
    V_sweep[base + k * n + q] = s * vkp + c * vkq;
}
