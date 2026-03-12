// SPDX-License-Identifier: AGPL-3.0-only
// Grid Quadrature GEMM (f64) — GPU Hamiltonian Construction
//
// Computes batched Hamiltonian matrices via numerical quadrature:
//   H[b,i,j] = Σ_k φ[b,i,k] * W[b,k] * φ[b,j,k] * quad_weights[k]
//
// This is the core operation for constructing Hamiltonian matrices
// from basis functions evaluated on a grid.
//
// Use cases:
//   - HFB Hamiltonian construction
//   - DFT matrix assembly
//   - Any basis function integral on a grid
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision
// - Zero unsafe code
// - Self-contained
//
// Algorithm:
// - Each workgroup computes one H[b,i,j] element
// - Parallel reduction over grid points within workgroup
// - Batched: multiple matrices in a single dispatch
//
// Memory layout (row-major):
//   phi[batch, n, grid] - basis functions on grid
//   w[batch, grid] - weight function (potential * r² etc.)
//   quad_weights[grid] - quadrature weights (shared across batch)
//   output[batch, n, n] - Hamiltonian matrices

struct QuadParams {
    batch_size: u32,    // Number of matrices to compute
    n: u32,             // Basis size (matrix dimension)
    grid_size: u32,     // Number of quadrature points
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> phi: array<f64>;           // [batch, n, grid]
@group(0) @binding(1) var<storage, read> w: array<f64>;             // [batch, grid]
@group(0) @binding(2) var<storage, read> quad_weights: array<f64>;  // [grid]
@group(0) @binding(3) var<storage, read_write> output: array<f64>;  // [batch, n, n]
@group(0) @binding(4) var<uniform> params: QuadParams;

// Shared memory for partial sums
var<workgroup> shared_sums: array<f64, 256>;

// Main kernel: each workgroup computes one H[b,i,j] element
// Dispatch: (n*n, batch_size, 1) with workgroup_size(256)
//
// Each workgroup:
// - Threads cooperatively sum over grid points
// - Final reduction produces H[b,i,j]
@compute @workgroup_size(256)
fn grid_quadrature_gemm(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let batch = wgid.y;
    let matrix_idx = wgid.x;  // Flattened (i, j) index
    
    if (batch >= params.batch_size) { return; }
    if (matrix_idx >= params.n * params.n) { return; }
    
    let i = matrix_idx / params.n;
    let j = matrix_idx % params.n;
    
    let n = params.n;
    let grid_size = params.grid_size;
    
    // Each thread processes a subset of grid points
    var local_sum = f64(0.0);
    var k = lid.x;
    
    while (k < grid_size) {
        // phi[batch, i, k] = phi[batch * n * grid + i * grid + k]
        let phi_i_k = phi[batch * n * grid_size + i * grid_size + k];
        let phi_j_k = phi[batch * n * grid_size + j * grid_size + k];
        let w_k = w[batch * grid_size + k];
        let qw_k = quad_weights[k];
        
        // Accumulate: φ_i * W * φ_j * quad_weight
        local_sum = local_sum + phi_i_k * w_k * phi_j_k * qw_k;
        
        k = k + 256u;  // Stride by workgroup size
    }
    
    // Store local sum in shared memory
    shared_sums[lid.x] = local_sum;
    workgroupBarrier();
    
    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            shared_sums[lid.x] = shared_sums[lid.x] + shared_sums[lid.x + stride];
        }
        workgroupBarrier();
    }
    
    // Write result
    if (lid.x == 0u) {
        output[batch * n * n + i * n + j] = shared_sums[0];
    }
}

// Optimized version for small grids (grid_size <= 256)
// Each thread handles one grid point, single reduction
@compute @workgroup_size(256)
fn grid_quadrature_gemm_small(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let batch = wgid.y;
    let matrix_idx = wgid.x;
    
    if (batch >= params.batch_size) { return; }
    if (matrix_idx >= params.n * params.n) { return; }
    
    let i = matrix_idx / params.n;
    let j = matrix_idx % params.n;
    
    let n = params.n;
    let grid_size = params.grid_size;
    
    // Each thread handles one grid point
    var value = f64(0.0);
    if (lid.x < grid_size) {
        let k = lid.x;
        let phi_i_k = phi[batch * n * grid_size + i * grid_size + k];
        let phi_j_k = phi[batch * n * grid_size + j * grid_size + k];
        let w_k = w[batch * grid_size + k];
        let qw_k = quad_weights[k];
        value = phi_i_k * w_k * phi_j_k * qw_k;
    }
    
    shared_sums[lid.x] = value;
    workgroupBarrier();
    
    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            shared_sums[lid.x] = shared_sums[lid.x] + shared_sums[lid.x + stride];
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        output[batch * n * n + i * n + j] = shared_sums[0];
    }
}

// Symmetric optimization: only compute upper triangle, then mirror
// Cuts computation in half for symmetric Hamiltonians
@compute @workgroup_size(256)
fn grid_quadrature_gemm_symmetric(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let batch = wgid.y;
    let upper_idx = wgid.x;  // Index into upper triangle
    
    if (batch >= params.batch_size) { return; }
    
    let n = params.n;
    let n_upper = (n * (n + 1u)) / 2u;  // Size of upper triangle
    
    if (upper_idx >= n_upper) { return; }
    
    // Convert linear upper-triangle index to (i, j)
    // Using formula: upper_idx = i*n - i*(i+1)/2 + j
    var i = 0u;
    var remaining = upper_idx;
    while (remaining >= n - i) {
        remaining = remaining - (n - i);
        i = i + 1u;
    }
    let j = i + remaining;
    
    let grid_size = params.grid_size;
    
    // Compute H[i,j]
    var local_sum = f64(0.0);
    var k = lid.x;
    
    while (k < grid_size) {
        let phi_i_k = phi[batch * n * grid_size + i * grid_size + k];
        let phi_j_k = phi[batch * n * grid_size + j * grid_size + k];
        let w_k = w[batch * grid_size + k];
        let qw_k = quad_weights[k];
        local_sum = local_sum + phi_i_k * w_k * phi_j_k * qw_k;
        k = k + 256u;
    }
    
    shared_sums[lid.x] = local_sum;
    workgroupBarrier();
    
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            shared_sums[lid.x] = shared_sums[lid.x] + shared_sums[lid.x + stride];
        }
        workgroupBarrier();
    }
    
    if (lid.x == 0u) {
        let value = shared_sums[0];
        // Write H[i,j]
        output[batch * n * n + i * n + j] = value;
        // Mirror to H[j,i] if i != j
        if (i != j) {
            output[batch * n * n + j * n + i] = value;
        }
    }
}
