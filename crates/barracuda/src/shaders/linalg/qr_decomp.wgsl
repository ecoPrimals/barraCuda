// SPDX-License-Identifier: AGPL-3.0-only
// QR Decomposition - Shader-First Implementation
//
// Householder QR factorization: A = QR
//
// For each column k:
// 1. Compute Householder vector v (parallel reduction for norm)
// 2. Apply H = I - 2vvᵀ to remaining columns (parallel matrix update)
// 3. Accumulate Q (optional, parallel)
//
// Sequential part (CPU):
//   - Column ordering (k = 0, 1, ..., n-1)
//
// Parallel part (GPU):
//   - Norm computation (parallel reduction)
//   - Householder application (O(n²) parallel per column)
//
// This is SHADER-FIRST QR:
// - Same Householder math as CPU
// - Each column application is massively parallel
// - Foundation for least squares, eigenvalue problems
//
// Reference: Golub & Van Loan, "Matrix Computations" (2013)

struct QrParams {
    m: u32,           // Rows
    n: u32,           // Columns
    k: u32,           // Current column
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: QrParams;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;    // Matrix [m × n], row-major
@group(0) @binding(2) var<storage, read_write> v: array<f32>;    // Householder vector [m]
@group(0) @binding(3) var<storage, read_write> tau: array<f32>;  // Householder scalars [min(m,n)]

var<workgroup> shared_sum: array<f32, 256>;

// Step 1: Compute column norm ||A[k:m, k]||
// Parallel reduction
@compute @workgroup_size(256, 1, 1)
fn column_norm(@builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(global_invocation_id) global_id: vec3<u32>,
               @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let m = params.m;
    let n = params.n;
    let k = params.k;
    
    // Sum of squares for rows k to m-1 in column k
    var sum_sq: f32 = 0.0;
    var i = k + tid;
    while (i < m) {
        let val = A[i * n + k];
        sum_sq = sum_sq + val * val;
        i = i + 256u;
    }
    
    shared_sum[tid] = sum_sq;
    workgroupBarrier();
    
    // Parallel reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        // Store norm squared in v[0] temporarily
        v[0] = shared_sum[0];
    }
}

// Step 2: Compute Householder vector v and scalar tau
// v = (A[k:m,k] - sign(A[k,k])*||A[k:m,k]||*e₁) / ||...||
// tau = 2 / (vᵀv)
struct HouseholderParams {
    m: u32,
    n: u32,
    k: u32,
    norm_sq: f32,     // ||A[k:m,k]||² from previous kernel
}

@group(0) @binding(0) var<uniform> hh_params: HouseholderParams;
@group(0) @binding(1) var<storage, read_write> A_hh: array<f32>;  // read_write for consistent bind group layout
@group(0) @binding(2) var<storage, read_write> v_hh: array<f32>;
@group(0) @binding(3) var<storage, read_write> tau_hh: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn compute_householder(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let m = hh_params.m;
    let n = hh_params.n;
    let k = hh_params.k;
    
    // Only process rows k to m-1
    let row = k + i;
    if (row >= m) {
        return;
    }
    
    let norm = sqrt(hh_params.norm_sq);
    let a_kk = A_hh[k * n + k];
    
    // sign(A[k,k]) * ||A[k:m,k]||
    var alpha: f32;
    if (a_kk >= 0.0) {
        alpha = -norm;
    } else {
        alpha = norm;
    }
    
    // v = A[k:m,k], then v[0] = v[0] - alpha
    if (row == k) {
        let v_0 = A_hh[row * n + k] - alpha;
        v_hh[i] = v_0;
        
        // Compute tau = 2 / (vᵀv)
        // vᵀv = (v[0])² + ||A[k+1:m,k]||²
        // ||A[k+1:m,k]||² = norm_sq - a_kk²
        let v_norm_sq = v_0 * v_0 + (hh_params.norm_sq - a_kk * a_kk);
        if (v_norm_sq > 1e-10) {
            tau_hh[k] = 2.0 / v_norm_sq;
        } else {
            tau_hh[k] = 0.0;
        }
    } else {
        v_hh[i] = A_hh[row * n + k];
    }
}

// Step 3: Apply Householder to remaining columns
// A[k:m, j] = A[k:m, j] - tau * v * (vᵀ * A[k:m, j])  for j > k
//
// Two sub-steps:
// 3a. Compute w[j] = vᵀ * A[k:m, j] for all j > k (parallel over j)
// 3b. Update A[i,j] -= tau * v[i] * w[j] (parallel over i,j)

struct ApplyParams {
    m: u32,
    n: u32,
    k: u32,
    tau_k: f32,       // tau[k] from previous kernel
}

@group(0) @binding(0) var<uniform> apply_params: ApplyParams;
@group(0) @binding(1) var<storage, read_write> v_apply: array<f32>;  // read_write for consistent bind group layout
@group(0) @binding(2) var<storage, read_write> A_apply: array<f32>;
@group(0) @binding(3) var<storage, read_write> w: array<f32>;  // Work array [n]

// Compute w[j] = vᵀ * A[k:m, j] for column j
// Each workgroup handles one column
@compute @workgroup_size(256, 1, 1)
fn compute_vTA(@builtin(local_invocation_id) local_id: vec3<u32>,
               @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let j = wg_id.x + apply_params.k + 1u;  // Column index (j > k)
    let m = apply_params.m;
    let n = apply_params.n;
    let k = apply_params.k;
    
    if (j >= n) {
        return;
    }
    
    // Dot product of v with column j
    var sum: f32 = 0.0;
    var i = tid;
    while (i < m - k) {
        let row = k + i;
        sum = sum + v_apply[i] * A_apply[row * n + j];
        i = i + 256u;
    }
    
    shared_sum[tid] = sum;
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    
    if (tid == 0u) {
        w[j] = shared_sum[0];
    }
}

// Update A[i,j] -= tau * v[i-k] * w[j] for i >= k, j > k
@compute @workgroup_size(16, 16, 1)
fn apply_householder(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rel_i = global_id.y;  // Relative row (0 = row k)
    let rel_j = global_id.x;  // Relative column (0 = column k+1)
    
    let m = apply_params.m;
    let n = apply_params.n;
    let k = apply_params.k;
    
    let i = k + rel_i;
    let j = k + 1u + rel_j;
    
    if (i >= m || j >= n) {
        return;
    }
    
    let tau_k = apply_params.tau_k;
    let v_i = v_apply[rel_i];
    let w_j = w[j];
    
    A_apply[i * n + j] = A_apply[i * n + j] - tau_k * v_i * w_j;
}

// Update column k (set to R values)
// A[k,k] = -sign(A[k,k]) * ||A[k:m,k]||, A[i,k] = 0 for i > k
@compute @workgroup_size(256, 1, 1)
fn update_column_k(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let m = apply_params.m;
    let n = apply_params.n;
    let k = apply_params.k;
    
    let row = k + i;
    if (row >= m) {
        return;
    }
    
    if (row > k) {
        // Zero below diagonal
        A_apply[row * n + k] = 0.0;
    }
    // Note: A[k,k] is updated separately or left as computed
}

// Accumulate Q = H₁ * H₂ * ... * Hₙ
// Q starts as identity, then Q = Q * Hₖ for each k
// Hₖ = I - tau[k] * vₖ * vₖᵀ
struct QAccumParams {
    m: u32,
    k: u32,
    tau_k: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> q_params: QAccumParams;
@group(0) @binding(1) var<storage, read_write> v_q: array<f32>;  // read_write for consistent bind group layout
@group(0) @binding(2) var<storage, read_write> Q: array<f32>;  // [m × m]

// Q = Q - tau * (Q * v) * vᵀ
// First compute w = Q * v (parallel), then update Q (parallel)
@compute @workgroup_size(256, 1, 1)
fn Q_times_v(@builtin(local_invocation_id) local_id: vec3<u32>,
             @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let i = wg_id.x;  // Row of Q
    let m = q_params.m;
    let k = q_params.k;
    
    if (i >= m) {
        return;
    }
    
    // w[i] = Σⱼ Q[i,j] * v[j-k] for j >= k
    var sum: f32 = 0.0;
    var j = k + tid;
    while (j < m) {
        sum = sum + Q[i * m + j] * v_q[j - k];
        j = j + 256u;
    }
    
    shared_sum[tid] = sum;
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }
    
    // Store in last element of row as temp
    if (tid == 0u) {
        Q[i * m + m - 1u] = shared_sum[0];  // Temporary storage
    }
}
