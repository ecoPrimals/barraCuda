// SPDX-License-Identifier: AGPL-3.0-only
// Cyclic Reduction (f64) — Parallel Tridiagonal Solver for PDEs
//
// Solves tridiagonal system: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
//
// **Deep Debt Evolution (Feb 16, 2026)**:
// - ✅ Pure WGSL f64 implementation
// - ✅ O(log n) parallel complexity (vs O(n) Thomas algorithm)
// - ✅ Hardware-agnostic (NVIDIA/AMD/Intel via WebGPU/Vulkan)
// - ✅ Science-grade precision for PDE stability
// - ✅ WGSL as unified math language
//
// Applications:
// - Crank-Nicolson PDE (heat, diffusion, Schrödinger)
// - Implicit finite difference schemes
// - Cubic spline interpolation
// - Poisson solvers
//
// Note: WGSL doesn't support `array<f64>` in workgroup memory,
// so this version uses global memory with careful synchronization.
// For small systems (n ≤ 128), the single-pass version is preferred.

struct Params {
    n: u32,           // System size
    step: u32,        // Current reduction step
    phase: u32,       // 0 = reduction, 1 = substitution
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> a: array<f64>;  // Sub-diagonal
@group(0) @binding(2) var<storage, read_write> b: array<f64>;  // Main diagonal
@group(0) @binding(3) var<storage, read_write> c: array<f64>;  // Super-diagonal
@group(0) @binding(4) var<storage, read_write> d: array<f64>;  // RHS / solution

// Helper for f64 constants
fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

// Reduction step: Eliminate odd-indexed unknowns
@compute @workgroup_size(256, 1, 1)
fn reduction_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let stride = 1u << (params.step + 1u);  // 2^(step+1)
    let half_stride = stride >> 1u;          // 2^step
    
    let i = global_id.x * stride + half_stride;  // Odd indices at this level
    
    if (i >= params.n) {
        return;
    }
    
    let i_prev = i - half_stride;
    let i_next = i + half_stride;
    
    if (i_next >= params.n) {
        return;
    }
    
    let pivot_ref = b[i];
    let epsilon = f64_const(pivot_ref, 1e-14);
    
    // Skip if diagonal is too small
    if (abs(b[i_prev]) < epsilon || abs(b[i_next]) < epsilon) {
        return;
    }
    
    // Compute elimination factors
    let alpha = -a[i] / b[i_prev];
    let gamma = -c[i] / b[i_next];
    
    // Update coefficients for the reduced system
    b[i] = b[i] + alpha * c[i_prev] + gamma * a[i_next];
    d[i] = d[i] + alpha * d[i_prev] + gamma * d[i_next];
    
    // Update connections to farther neighbors
    a[i] = alpha * a[i_prev];
    c[i] = gamma * c[i_next];
}

// Substitution step: Back-substitute to recover solution
@compute @workgroup_size(256, 1, 1)
fn substitution_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let stride = 1u << (params.step + 1u);
    let half_stride = stride >> 1u;
    
    let i = global_id.x * stride + half_stride;
    
    if (i >= params.n) {
        return;
    }
    
    let i_prev = i - half_stride;
    let i_next = i + half_stride;
    
    let pivot_ref = b[i];
    let zero = f64_const(pivot_ref, 0.0);
    let epsilon = f64_const(pivot_ref, 1e-14);
    
    var x_prev = zero;
    var x_next = zero;
    
    if (i_prev < params.n) {
        x_prev = d[i_prev];
    }
    if (i_next < params.n) {
        x_next = d[i_next];
    }
    
    if (abs(b[i]) > epsilon) {
        d[i] = (d[i] - a[i] * x_prev - c[i] * x_next) / b[i];
    }
}

// =============================================================================
// SINGLE-THREAD SOLVER: For small systems, serial is faster than sync overhead
// =============================================================================

@compute @workgroup_size(1)
fn solve_serial_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x != 0u) {
        return;
    }
    
    let n = params.n;
    let pivot_ref = b[0];
    let epsilon = f64_const(pivot_ref, 1e-14);
    let zero = f64_const(pivot_ref, 0.0);
    
    // Thomas algorithm (serial, but optimal for small n)
    // Forward elimination
    for (var i = 1u; i < n; i = i + 1u) {
        if (abs(b[i - 1u]) < epsilon) {
            return;  // Singular
        }
        let m = a[i] / b[i - 1u];
        b[i] = b[i] - m * c[i - 1u];
        d[i] = d[i] - m * d[i - 1u];
    }
    
    // Back substitution
    if (abs(b[n - 1u]) < epsilon) {
        return;
    }
    d[n - 1u] = d[n - 1u] / b[n - 1u];
    
    for (var i_rev = 1u; i_rev < n; i_rev = i_rev + 1u) {
        let i = n - 1u - i_rev;
        if (abs(b[i]) < epsilon) {
            return;
        }
        d[i] = (d[i] - c[i] * d[i + 1u]) / b[i];
    }
}

// =============================================================================
// BATCHED SOLVER: Multiple independent systems
// =============================================================================

struct BatchParams {
    n: u32,           // System size per batch
    n_padded: u32,    // Padded size (power of 2)
    batch_size: u32,  // Number of systems
    step: u32,        // Current reduction step
}

@group(1) @binding(0) var<uniform> batch_params: BatchParams;

// Batched reduction step
@compute @workgroup_size(256, 1, 1)
fn reduction_batch_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                       @builtin(workgroup_id) wg_id: vec3<u32>) {
    let batch_idx = wg_id.y;
    let n_padded = batch_params.n_padded;
    let step = batch_params.step;
    
    if (batch_idx >= batch_params.batch_size) {
        return;
    }
    
    let offset = batch_idx * n_padded;
    let stride = 1u << (step + 1u);
    let half_stride = stride >> 1u;
    
    let i = global_id.x * stride + half_stride;
    
    if (i >= batch_params.n) {
        return;
    }
    
    let i_prev = i - half_stride;
    let i_next = i + half_stride;
    
    if (i_next >= batch_params.n) {
        return;
    }
    
    let idx = offset + i;
    let idx_prev = offset + i_prev;
    let idx_next = offset + i_next;
    
    let pivot_ref = b[idx];
    let epsilon = f64_const(pivot_ref, 1e-14);
    
    if (abs(b[idx_prev]) < epsilon || abs(b[idx_next]) < epsilon) {
        return;
    }
    
    let alpha = -a[idx] / b[idx_prev];
    let gamma = -c[idx] / b[idx_next];
    
    b[idx] = b[idx] + alpha * c[idx_prev] + gamma * a[idx_next];
    d[idx] = d[idx] + alpha * d[idx_prev] + gamma * d[idx_next];
    a[idx] = alpha * a[idx_prev];
    c[idx] = gamma * c[idx_next];
}

// Batched substitution step
@compute @workgroup_size(256, 1, 1)
fn substitution_batch_f64(@builtin(global_invocation_id) global_id: vec3<u32>,
                          @builtin(workgroup_id) wg_id: vec3<u32>) {
    let batch_idx = wg_id.y;
    let n_padded = batch_params.n_padded;
    let step = batch_params.step;
    
    if (batch_idx >= batch_params.batch_size) {
        return;
    }
    
    let offset = batch_idx * n_padded;
    let stride = 1u << (step + 1u);
    let half_stride = stride >> 1u;
    
    let i = global_id.x * stride + half_stride;
    
    if (i >= batch_params.n) {
        return;
    }
    
    let i_prev = i - half_stride;
    let i_next = i + half_stride;
    let idx = offset + i;
    
    let pivot_ref = b[idx];
    let zero = f64_const(pivot_ref, 0.0);
    let epsilon = f64_const(pivot_ref, 1e-14);
    
    var x_prev = zero;
    var x_next = zero;
    
    if (i_prev < batch_params.n) {
        x_prev = d[offset + i_prev];
    }
    if (i_next < batch_params.n) {
        x_next = d[offset + i_next];
    }
    
    if (abs(b[idx]) > epsilon) {
        d[idx] = (d[idx] - a[idx] * x_prev - c[idx] * x_next) / b[idx];
    }
}

// Batched serial solver (Thomas algorithm) for small systems
// One workgroup per system
@compute @workgroup_size(1)
fn solve_batch_serial_f64(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let batch_idx = wg_id.x;
    
    if (batch_idx >= batch_params.batch_size) {
        return;
    }
    
    let n = batch_params.n;
    let offset = batch_idx * batch_params.n_padded;
    
    let pivot_ref = b[offset];
    let epsilon = f64_const(pivot_ref, 1e-14);
    
    // Thomas algorithm
    // Forward elimination
    for (var i = 1u; i < n; i = i + 1u) {
        let idx = offset + i;
        let idx_prev = offset + i - 1u;
        
        if (abs(b[idx_prev]) < epsilon) {
            return;
        }
        let m = a[idx] / b[idx_prev];
        b[idx] = b[idx] - m * c[idx_prev];
        d[idx] = d[idx] - m * d[idx_prev];
    }
    
    // Back substitution
    let last = offset + n - 1u;
    if (abs(b[last]) < epsilon) {
        return;
    }
    d[last] = d[last] / b[last];
    
    for (var i_rev = 1u; i_rev < n; i_rev = i_rev + 1u) {
        let i = n - 1u - i_rev;
        let idx = offset + i;
        
        if (abs(b[idx]) < epsilon) {
            return;
        }
        d[idx] = (d[idx] - c[idx] * d[offset + i + 1u]) / b[idx];
    }
}
