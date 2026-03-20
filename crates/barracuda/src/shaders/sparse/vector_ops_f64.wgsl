// SPDX-License-Identifier: AGPL-3.0-or-later
// Vector operations for sparse iterative solvers - f64 Precision
// All operations use consistent read/read_write bindings

// ============================================================================
// AXPY: y = alpha * x + y
// ============================================================================
struct AxpyParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f64,
}

@group(0) @binding(0) var<storage, read> axpy_x: array<f64>;
@group(0) @binding(1) var<storage, read_write> axpy_y: array<f64>;
@group(0) @binding(2) var<uniform> axpy_params: AxpyParams;

@compute @workgroup_size(256)
fn axpy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= axpy_params.n) {
        return;
    }
    axpy_y[idx] = axpy_params.alpha * axpy_x[idx] + axpy_y[idx];
}

// ============================================================================
// Scale: y = alpha * x
// ============================================================================
struct ScaleParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f64,
}

@group(0) @binding(0) var<storage, read> scale_x: array<f64>;
@group(0) @binding(1) var<storage, read_write> scale_y: array<f64>;
@group(0) @binding(2) var<uniform> scale_params: ScaleParams;

@compute @workgroup_size(256)
fn scale_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scale_params.n) {
        return;
    }
    scale_y[idx] = scale_params.alpha * scale_x[idx];
}

// ============================================================================
// Copy: y = x
// ============================================================================
struct CopyParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> copy_src: array<f64>;
@group(0) @binding(1) var<storage, read_write> copy_dst: array<f64>;
@group(0) @binding(2) var<uniform> copy_params: CopyParams;

@compute @workgroup_size(256)
fn copy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= copy_params.n) {
        return;
    }
    copy_dst[idx] = copy_src[idx];
}

// ============================================================================
// Linear combination: z = alpha * x + beta * y
// ============================================================================
struct LCParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f64,
    beta: f64,
}

@group(0) @binding(0) var<storage, read> lc_x: array<f64>;
@group(0) @binding(1) var<storage, read> lc_y: array<f64>;
@group(0) @binding(2) var<storage, read_write> lc_z: array<f64>;
@group(0) @binding(3) var<uniform> lc_params: LCParams;

@compute @workgroup_size(256)
fn linear_comb_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= lc_params.n) {
        return;
    }
    lc_z[idx] = lc_params.alpha * lc_x[idx] + lc_params.beta * lc_y[idx];
}

// ============================================================================
// Diagonal preconditioner: z[i] = r[i] / diag[i]
// ============================================================================
struct PrecondParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> precond_r: array<f64>;
@group(0) @binding(1) var<storage, read> precond_diag: array<f64>;
@group(0) @binding(2) var<storage, read_write> precond_z: array<f64>;
@group(0) @binding(3) var<uniform> precond_params: PrecondParams;

@compute @workgroup_size(256)
fn precond_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= precond_params.n) {
        return;
    }
    let d = precond_diag[idx];
    // Safe division - avoid very small divisors
    if (abs(d) > f64(1e-12)) {
        precond_z[idx] = precond_r[idx] / d;
    } else {
        precond_z[idx] = precond_r[idx];
    }
}
