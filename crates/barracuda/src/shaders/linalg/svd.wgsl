// SPDX-License-Identifier: AGPL-3.0-or-later
// SVD (Singular Value Decomposition) - Shader-First Implementation
//
// Computes A = U Σ Vᵀ where:
//   - U is m×m orthogonal (left singular vectors)
//   - Σ is m×n diagonal (singular values)
//   - Vᵀ is n×n orthogonal (right singular vectors)
//
// Algorithm: One-sided Jacobi SVD
// 1. Compute B = AᵀA (parallel matmul)
// 2. Jacobi eigendecomp of B gives V and σ² (uses eigh.wgsl pattern)
// 3. U = A V Σ⁻¹ (parallel matmul)
//
// Sequential part (CPU):
//   - Jacobi sweep ordering (convergence loop)
//   - Rotation pair selection
//
// Parallel part (GPU):
//   - AᵀA computation (O(mn²) parallel)
//   - Jacobi rotation application (O(n²) per rotation, parallel)
//   - U computation (O(mn) parallel)
//
// This is SHADER-FIRST SVD:
// - Jacobi method is numerically stable
// - Each rotation is a parallel 2×2 transform across all rows
// - Foundation for low-rank approximation, pseudoinverse
//
// Reference: Demmel & Veselic (1992), "Jacobi's Method is More Accurate than QR"

struct SvdParams {
    m: u32,           // Rows of A
    n: u32,           // Columns of A
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: SvdParams;
@group(0) @binding(1) var<storage, read> A: array<f32>;         // Input [m × n]
@group(0) @binding(2) var<storage, read_write> B: array<f32>;   // AᵀA [n × n]
@group(0) @binding(3) var<storage, read_write> V: array<f32>;   // Right singular vectors [n × n]
@group(0) @binding(4) var<storage, read_write> sigma: array<f32>; // Singular values [n]

// Compute B = AᵀA - parallel over (i,j)
@compute @workgroup_size(16, 16, 1)
fn compute_AtA(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;  // Row of B
    let j = global_id.x;  // Column of B
    let m = params.m;
    let n = params.n;
    
    if (i >= n || j >= n) {
        return;
    }
    
    // B[i,j] = Σₖ A[k,i] * A[k,j]
    var sum: f32 = 0.0;
    for (var k = 0u; k < m; k = k + 1u) {
        sum = sum + A[k * n + i] * A[k * n + j];
    }
    
    B[i * n + j] = sum;
}

// Initialize V = I
@compute @workgroup_size(16, 16, 1)
fn init_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;
    let j = global_id.x;
    let n = params.n;
    
    if (i >= n || j >= n) {
        return;
    }
    
    if (i == j) {
        V[i * n + j] = 1.0;
    } else {
        V[i * n + j] = 0.0;
    }
}

// Jacobi rotation parameters for a 2×2 symmetric matrix
// Given [[a, b], [b, c]], compute cos(θ) and sin(θ) for diagonalization
struct JacobiRotation {
    p: u32,           // First index
    q: u32,           // Second index (q > p)
    c: f32,           // cos(θ)
    s: f32,           // sin(θ)
}

// Apply Jacobi rotation to B (symmetric): B = JᵀBJ
// Only affects rows/columns p and q
struct RotationParams {
    n: u32,
    p: u32,
    q: u32,
    c: f32,           // cos(θ)
}

@group(0) @binding(0) var<uniform> rot_params: RotationParams;
@group(0) @binding(1) var<storage, read_write> B_rot: array<f32>;
@group(0) @binding(2) var<storage, read> s_val: array<f32>;  // sin(θ) passed separately

// Update row p and q of B
@compute @workgroup_size(256, 1, 1)
fn jacobi_rotate_B(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let n = rot_params.n;
    let p = rot_params.p;
    let q = rot_params.q;
    let c = rot_params.c;
    let s = s_val[0];
    
    if (j >= n || j == p || j == q) {
        return;
    }
    
    // B'[p,j] = c*B[p,j] + s*B[q,j]
    // B'[q,j] = -s*B[p,j] + c*B[q,j]
    let B_pj = B_rot[p * n + j];
    let B_qj = B_rot[q * n + j];
    
    B_rot[p * n + j] = c * B_pj + s * B_qj;
    B_rot[q * n + j] = -s * B_pj + c * B_qj;
    
    // Symmetric: also update columns
    B_rot[j * n + p] = B_rot[p * n + j];
    B_rot[j * n + q] = B_rot[q * n + j];
}

// Update the 2×2 block B[p,q] - makes it diagonal
@compute @workgroup_size(1, 1, 1)
fn jacobi_update_block() {
    let n = rot_params.n;
    let p = rot_params.p;
    let q = rot_params.q;
    let c = rot_params.c;
    let s = s_val[0];
    
    let B_pp = B_rot[p * n + p];
    let B_pq = B_rot[p * n + q];
    let B_qq = B_rot[q * n + q];
    
    // After rotation, the (p,q) block becomes diagonal
    // B'[p,p] = c²B[p,p] + 2cs·B[p,q] + s²B[q,q]
    // B'[q,q] = s²B[p,p] - 2cs·B[p,q] + c²B[q,q]
    // B'[p,q] = 0 (by construction)
    
    B_rot[p * n + p] = c*c*B_pp + 2.0*c*s*B_pq + s*s*B_qq;
    B_rot[q * n + q] = s*s*B_pp - 2.0*c*s*B_pq + c*c*B_qq;
    B_rot[p * n + q] = 0.0;
    B_rot[q * n + p] = 0.0;
}

// Apply Jacobi rotation to V: V = V * J
// Each column of V is updated
@compute @workgroup_size(256, 1, 1)
fn jacobi_rotate_V(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;  // Row index
    let n = rot_params.n;
    let p = rot_params.p;
    let q = rot_params.q;
    let c = rot_params.c;
    let s = s_val[0];
    
    if (i >= n) {
        return;
    }
    
    // V'[i,p] = c*V[i,p] + s*V[i,q]
    // V'[i,q] = -s*V[i,p] + c*V[i,q]
    let V_ip = V[i * n + p];
    let V_iq = V[i * n + q];
    
    V[i * n + p] = c * V_ip + s * V_iq;
    V[i * n + q] = -s * V_ip + c * V_iq;
}

// Extract singular values from diagonal of B
// σᵢ = √B[i,i]
@compute @workgroup_size(256, 1, 1)
fn extract_sigma(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    
    if (i >= n) {
        return;
    }
    
    let diag = B[i * n + i];
    if (diag >= 0.0) {
        sigma[i] = sqrt(diag);
    } else {
        sigma[i] = 0.0;  // Numerical noise
    }
}

// Compute U = A * V * Σ⁻¹
// U[:,i] = (1/σᵢ) * A * V[:,i]
struct UParams {
    m: u32,
    n: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> u_params: UParams;
@group(0) @binding(1) var<storage, read> A_u: array<f32>;
@group(0) @binding(2) var<storage, read> V_u: array<f32>;
@group(0) @binding(3) var<storage, read> sigma_u: array<f32>;
@group(0) @binding(4) var<storage, read_write> U: array<f32>;  // [m × n]

@compute @workgroup_size(16, 16, 1)
fn compute_U(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;  // Row of U
    let j = global_id.x;  // Column of U (= singular vector index)
    let m = u_params.m;
    let n = u_params.n;
    
    if (i >= m || j >= n) {
        return;
    }
    
    let sig = sigma_u[j];
    if (sig < 1e-10) {
        U[i * n + j] = 0.0;
        return;
    }
    
    // U[i,j] = (1/σⱼ) Σₖ A[i,k] * V[k,j]
    var sum: f32 = 0.0;
    for (var k = 0u; k < n; k = k + 1u) {
        sum = sum + A_u[i * n + k] * V_u[k * n + j];
    }
    
    U[i * n + j] = sum / sig;
}

// Compute off-diagonal norm (for convergence check)
// ||off(B)||_F² = Σᵢ<ⱼ B[i,j]²
struct NormParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> norm_params: NormParams;
@group(0) @binding(1) var<storage, read> B_norm: array<f32>;
@group(0) @binding(2) var<storage, read_write> off_norm: array<f32>;

var<workgroup> shared_norm: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn off_diagonal_norm(@builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = local_id.x;
    let n = norm_params.n;
    
    // Each thread sums its portion of off-diagonal elements
    var sum: f32 = 0.0;
    let num_pairs = n * (n - 1u) / 2u;
    
    var pair_idx = tid;
    while (pair_idx < num_pairs) {
        // Convert linear index to (i,j) with i < j
        // pair_idx = i*n - i*(i+1)/2 + j - i - 1
        // Solve for i,j (simpler: iterate)
        var i = 0u;
        var count = 0u;
        for (var ii = 0u; ii < n - 1u; ii = ii + 1u) {
            let pairs_in_row = n - 1u - ii;
            if (count + pairs_in_row > pair_idx) {
                i = ii;
                break;
            }
            count = count + pairs_in_row;
        }
        let j = pair_idx - count + i + 1u;
        
        let val = B_norm[i * n + j];
        sum = sum + val * val;
        
        pair_idx = pair_idx + 256u;
    }
    
    shared_norm[tid] = sum;
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_norm[tid] = shared_norm[tid] + shared_norm[tid + stride];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        off_norm[0] = shared_norm[0];
    }
}

// Find maximum off-diagonal element (for classical Jacobi)
@compute @workgroup_size(256, 1, 1)
fn find_max_off_diag(@builtin(local_invocation_id) local_id: vec3<u32>,
                     @builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = local_id.x;
    let n = norm_params.n;
    
    var max_val: f32 = 0.0;
    var max_p: u32 = 0u;
    var max_q: u32 = 1u;
    
    let num_pairs = n * (n - 1u) / 2u;
    var pair_idx = tid;
    
    while (pair_idx < num_pairs) {
        // Same indexing as above
        var i = 0u;
        var count = 0u;
        for (var ii = 0u; ii < n - 1u; ii = ii + 1u) {
            let pairs_in_row = n - 1u - ii;
            if (count + pairs_in_row > pair_idx) {
                i = ii;
                break;
            }
            count = count + pairs_in_row;
        }
        let j = pair_idx - count + i + 1u;
        
        let val = abs(B_norm[i * n + j]);
        if (val > max_val) {
            max_val = val;
            max_p = i;
            max_q = j;
        }
        
        pair_idx = pair_idx + 256u;
    }
    
    // Store in shared memory for reduction
    shared_norm[tid] = max_val;
    // Would need additional shared arrays for indices - simplified here
    workgroupBarrier();
    
    // Parallel max reduction (simplified - full impl needs index tracking)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            if (shared_norm[tid + stride] > shared_norm[tid]) {
                shared_norm[tid] = shared_norm[tid + stride];
            }
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        off_norm[0] = shared_norm[0];  // Max value
    }
}
