// SPDX-License-Identifier: AGPL-3.0-or-later
// Symmetric Eigenvalue Decomposition (f64) - Shader-First Implementation
//
// Full f64 precision via WGSL native f64 support (SPIR-V/Vulkan)
// FP64 performance: 1:2-3 (not 1:32 like CUDA consumer cards)
//
// Jacobi eigenvalue algorithm for symmetric matrices:
// A = V · D · Vᵀ where D is diagonal (eigenvalues) and V is orthogonal (eigenvectors)
//
// Multi-pass algorithm:
// 1. init_V:        V = I (identity matrix)
// 2. find_max_off:  Find largest off-diagonal |A[p,q]|
// 3. jacobi_angle:  Compute rotation angle θ from A[p,p], A[q,q], A[p,q]
// 4. jacobi_rotate_A: Apply rotation to A (rows/cols p,q)
// 5. jacobi_rotate_V: Apply rotation to V (rows/cols p,q)
// 6. extract_eigenvalues: Read diagonal of A as eigenvalues
//
// References:
// - Golub & Van Loan, "Matrix Computations", Section 8.4
// - Demmel & Veselic (1992), "Jacobi's Method is More Accurate than QR"

struct EighParams {
    n: u32,           // Matrix dimension
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: EighParams;
@group(0) @binding(1) var<storage, read_write> A: array<f64>;    // Symmetric matrix [n × n], row-major
@group(0) @binding(2) var<storage, read_write> V: array<f64>;    // Eigenvectors [n × n], row-major
@group(0) @binding(3) var<storage, read_write> eigenvalues: array<f64>;  // Eigenvalues [n]

// Initialize V = Identity matrix
@compute @workgroup_size(16, 16, 1)
fn init_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;
    let n = params.n;

    if (row >= n || col >= n) {
        return;
    }

    let idx = row * n + col;
    // Naga resolves 1.0 as f32; use f64() constructor for storage writes
    if (row == col) {
        V[idx] = f64(1.0);
    } else {
        V[idx] = f64(0.0);
    }
}

// Jacobi rotation parameters
struct RotParams {
    n: u32,
    p: u32,           // First pivot index
    q: u32,           // Second pivot index (q > p)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> rot_params: RotParams;
@group(0) @binding(1) var<storage, read_write> A_rot: array<f64>;
@group(0) @binding(2) var<storage, read_write> cs: array<f64>;   // [c, s] = [cos(θ), sin(θ)]

// Compute Jacobi rotation angle for indices (p, q)
// Outputs c = cos(θ), s = sin(θ) to cs buffer
@compute @workgroup_size(1, 1, 1)
fn compute_jacobi_angle(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = rot_params.n;
    let p = rot_params.p;
    let q = rot_params.q;

    let app = A_rot[p * n + p];
    let aqq = A_rot[q * n + q];
    let apq = A_rot[p * n + q];

    // Handle near-zero off-diagonal
    if (abs(apq) < 1e-14) {
        cs[0] = f64(1.0);   // c = 1
        cs[1] = f64(0.0);   // s = 0
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
        // t = sign(phi) / (|phi| + sqrt(1 + phi²))
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

    cs[0] = c;
    cs[1] = s;
}

// Apply Jacobi rotation to matrix A: rows and columns p, q
// Uses cs buffer for cos/sin values
@group(0) @binding(0) var<uniform> apply_params: RotParams;
@group(0) @binding(1) var<storage, read_write> A_apply: array<f64>;
@group(0) @binding(2) var<storage, read> cs_apply: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn jacobi_rotate_A(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let n = apply_params.n;
    let p = apply_params.p;
    let q = apply_params.q;

    if (k >= n) {
        return;
    }

    let c = cs_apply[0];
    let s = cs_apply[1];

    // Skip the (p,q) element - handled separately
    if (k == p || k == q) {
        return;
    }

    // Update row k (columns p and q): A[k,p] and A[k,q]
    let akp = A_apply[k * n + p];
    let akq = A_apply[k * n + q];

    A_apply[k * n + p] = c * akp - s * akq;
    A_apply[k * n + q] = s * akp + c * akq;

    // Update column k (rows p and q): A[p,k] and A[q,k] - symmetric
    A_apply[p * n + k] = c * akp - s * akq;
    A_apply[q * n + k] = s * akp + c * akq;
}

// Update the 2×2 block at (p,p), (p,q), (q,p), (q,q)
@compute @workgroup_size(1, 1, 1)
fn jacobi_update_block(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let n = apply_params.n;
    let p = apply_params.p;
    let q = apply_params.q;

    let c = cs_apply[0];
    let s = cs_apply[1];

    let app = A_apply[p * n + p];
    let aqq = A_apply[q * n + q];
    let apq = A_apply[p * n + q];

    // New diagonal elements after rotation
    let app_new = c * c * app - 2.0 * c * s * apq + s * s * aqq;
    let aqq_new = s * s * app + 2.0 * c * s * apq + c * c * aqq;

    A_apply[p * n + p] = app_new;
    A_apply[q * n + q] = aqq_new;

    // Off-diagonal should be zero after rotation
    A_apply[p * n + q] = f64(0.0);
    A_apply[q * n + p] = f64(0.0);
}

// Apply Jacobi rotation to eigenvector matrix V: columns p and q
@group(0) @binding(0) var<uniform> v_params: RotParams;
@group(0) @binding(1) var<storage, read_write> V_rot: array<f64>;
@group(0) @binding(2) var<storage, read> cs_v: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn jacobi_rotate_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let k = global_id.x;
    let n = v_params.n;
    let p = v_params.p;
    let q = v_params.q;

    if (k >= n) {
        return;
    }

    let c = cs_v[0];
    let s = cs_v[1];

    // Update row k: V[k,p] and V[k,q]
    let vkp = V_rot[k * n + p];
    let vkq = V_rot[k * n + q];

    V_rot[k * n + p] = c * vkp - s * vkq;
    V_rot[k * n + q] = s * vkp + c * vkq;
}

// Extract eigenvalues from diagonal of A
@group(0) @binding(0) var<uniform> extract_params: EighParams;
@group(0) @binding(1) var<storage, read> A_extract: array<f64>;
@group(0) @binding(2) var<storage, read_write> eig_out: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn extract_eigenvalues(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = extract_params.n;

    if (i >= n) {
        return;
    }

    eig_out[i] = A_extract[i * n + i];
}

// Find maximum off-diagonal element (reduction)
// Each workgroup finds its local max, writes to partial_max buffer
@group(0) @binding(0) var<uniform> max_params: EighParams;
@group(0) @binding(1) var<storage, read> A_max: array<f64>;
@group(0) @binding(2) var<storage, read_write> partial_max: array<f64>;  // [max_val, p, q] per workgroup

var<workgroup> shared_max: array<f64, 256>;
var<workgroup> shared_p: array<u32, 256>;
var<workgroup> shared_q: array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn find_max_off_diag(@builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(global_invocation_id) global_id: vec3<u32>,
                     @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let n = max_params.n;

    // Total off-diagonal elements (upper triangle): n*(n-1)/2
    let total_pairs = (n * (n - 1u)) / 2u;

    // Each thread handles multiple pairs
    var local_max: f64 = f64(0.0);
    var local_p: u32 = 0u;
    var local_q: u32 = 1u;

    var pair_idx = global_id.x;
    while (pair_idx < total_pairs) {
        // Convert linear index to (p, q) where p < q
        // Using quadratic formula to find p from linear index
        var p: u32 = 0u;
        var offset: u32 = 0u;
        for (var i = 0u; i < n - 1u; i = i + 1u) {
            let pairs_in_row = n - 1u - i;
            if (pair_idx < offset + pairs_in_row) {
                p = i;
                break;
            }
            offset = offset + pairs_in_row;
        }
        let q = pair_idx - offset + p + 1u;

        if (p < n && q < n) {
            let val = abs(A_max[p * n + q]);
            if (val > local_max) {
                local_max = val;
                local_p = p;
                local_q = q;
            }
        }

        pair_idx = pair_idx + 256u * 256u;  // Stride by total threads
    }

    shared_max[tid] = local_max;
    shared_p[tid] = local_p;
    shared_q[tid] = local_q;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            if (shared_max[tid + stride] > shared_max[tid]) {
                shared_max[tid] = shared_max[tid + stride];
                shared_p[tid] = shared_p[tid + stride];
                shared_q[tid] = shared_q[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let out_idx = wg_id.x * 3u;
        partial_max[out_idx] = shared_max[0];
        partial_max[out_idx + 1u] = f64(shared_p[0]);
        partial_max[out_idx + 2u] = f64(shared_q[0]);
    }
}
