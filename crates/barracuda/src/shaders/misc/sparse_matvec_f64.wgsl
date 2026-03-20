// SPDX-License-Identifier: AGPL-3.0-or-later
// Sparse Matrix-Vector Product (CSR format) - f64 Precision
// y = A * x where A is in Compressed Sparse Row format
//
// Full f64 precision via WGSL native f64 support (SPIR-V/Vulkan)
// FP64 performance: 1:2-3 (not 1:32 like CUDA consumer cards)
//
// Uses atomic-free design: one thread per row, full precision

struct Params {
    num_rows: u32,
}

@group(0) @binding(0) var<storage, read> values: array<f64>;
@group(0) @binding(1) var<storage, read> col_indices: array<u32>;
@group(0) @binding(2) var<storage, read> row_ptrs: array<u32>;
@group(0) @binding(3) var<storage, read> vector: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn spmv_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= params.num_rows) {
        return;
    }

    let start = row_ptrs[row];
    let end = row_ptrs[row + 1u];

    var sum: f64 = 0.0;
    for (var j = start; j < end; j = j + 1u) {
        let col = col_indices[j];
        let val = values[j];
        sum = sum + val * vector[col];
    }
    output[row] = sum;
}

// Vector operations for iterative solvers

// y = alpha * x + y (axpy)
@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read_write> y_axpy: array<f64>;
@group(0) @binding(2) var<uniform> axpy_params: AxpyParams;

struct AxpyParams {
    n: u32,
    alpha: f64,
}

@compute @workgroup_size(256)
fn axpy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= axpy_params.n) {
        return;
    }
    y_axpy[idx] = axpy_params.alpha * x[idx] + y_axpy[idx];
}

// Dot product reduction (per-workgroup partial sum)
@group(0) @binding(0) var<storage, read> dot_a: array<f64>;
@group(0) @binding(1) var<storage, read> dot_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f64>;
@group(0) @binding(3) var<uniform> dot_params: DotParams;

struct DotParams {
    n: u32,
}

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn dot_f64(@builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = dot_params.n;

    // Each thread sums its elements
    var sum: f64 = 0.0;
    var i = gid;
    while (i < n) {
        sum = sum + dot_a[i] * dot_b[i];
        i = i + 256u * 256u;  // Stride by total threads
    }

    shared_sum[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        partial_sums[wg_id.x] = shared_sum[0];
    }
}

// Scale vector: y = alpha * x
@group(0) @binding(0) var<storage, read> scale_x: array<f64>;
@group(0) @binding(1) var<storage, read_write> scale_y: array<f64>;
@group(0) @binding(2) var<uniform> scale_params: ScaleParams;

struct ScaleParams {
    n: u32,
    alpha: f64,
}

@compute @workgroup_size(256)
fn scale_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= scale_params.n) {
        return;
    }
    scale_y[idx] = scale_params.alpha * scale_x[idx];
}

// Copy vector: y = x
@group(0) @binding(0) var<storage, read> copy_src: array<f64>;
@group(0) @binding(1) var<storage, read_write> copy_dst: array<f64>;
@group(0) @binding(2) var<uniform> copy_params: CopyParams;

struct CopyParams {
    n: u32,
}

@compute @workgroup_size(256)
fn copy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= copy_params.n) {
        return;
    }
    copy_dst[idx] = copy_src[idx];
}

// Apply diagonal preconditioner: z[i] = r[i] / diag[i]
@group(0) @binding(0) var<storage, read> r_precond: array<f64>;
@group(0) @binding(1) var<storage, read> diag_precond: array<f64>;
@group(0) @binding(2) var<storage, read_write> z_precond: array<f64>;
@group(0) @binding(3) var<uniform> precond_params: PrecondParams;

struct PrecondParams {
    n: u32,
}

@compute @workgroup_size(256)
fn precond_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= precond_params.n) {
        return;
    }
    let d = diag_precond[idx];
    // Safe division - avoid very small divisors
    if (abs(d) > 1e-12) {
        z_precond[idx] = r_precond[idx] / d;
    } else {
        z_precond[idx] = r_precond[idx];
    }
}

// Linear combination: z = alpha * x + beta * y
@group(0) @binding(0) var<storage, read> lc_x: array<f64>;
@group(0) @binding(1) var<storage, read> lc_y: array<f64>;
@group(0) @binding(2) var<storage, read_write> lc_z: array<f64>;
@group(0) @binding(3) var<uniform> lc_params: LCParams;

struct LCParams {
    n: u32,
    alpha: f64,
    beta: f64,
}

@compute @workgroup_size(256)
fn linear_comb_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= lc_params.n) {
        return;
    }
    lc_z[idx] = lc_params.alpha * lc_x[idx] + lc_params.beta * lc_y[idx];
}

// ============================================================================
// GPU-Resident CG Operations (Feb 14 2026)
// 
// These kernels keep scalar values on GPU to eliminate per-iteration CPU sync.
// hotSpring recommendation: only read back residual every N iterations.
// ============================================================================

// Final reduction: sum partial_sums[0..n_workgroups] into scalar_result[0]
@group(0) @binding(0) var<storage, read> partial_sums_final: array<f64>;
@group(0) @binding(1) var<storage, read_write> scalar_result: array<f64>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

struct ReduceParams {
    n_workgroups: u32,
}

var<workgroup> final_shared: array<f64, 256>;

@compute @workgroup_size(256)
fn final_reduce_f64(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = reduce_params.n_workgroups;

    // Load partial sums (handle case where n < 256)
    var sum: f64 = 0.0;
    if (tid < n) {
        sum = partial_sums_final[tid];
    }

    // For n > 256, each thread sums multiple elements
    var i = tid + 256u;
    while (i < n) {
        sum = sum + partial_sums_final[i];
        i = i + 256u;
    }

    final_shared[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            final_shared[tid] = final_shared[tid] + final_shared[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        scalar_result[0] = final_shared[0];
    }
}

// CG update step 1: x = x + alpha * p (alpha is a scalar on GPU)
// r = r - alpha * Ap
@group(0) @binding(0) var<storage, read_write> cg_x: array<f64>;
@group(0) @binding(1) var<storage, read_write> cg_r: array<f64>;
@group(0) @binding(2) var<storage, read> cg_p: array<f64>;
@group(0) @binding(3) var<storage, read> cg_Ap: array<f64>;
@group(0) @binding(4) var<storage, read> cg_alpha: array<f64>;  // alpha[0]
@group(0) @binding(5) var<uniform> cg_params1: CGParams;

struct CGParams {
    n: u32,
}

@compute @workgroup_size(256)
fn cg_update_xr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cg_params1.n) {
        return;
    }

    let alpha = cg_alpha[0];
    cg_x[idx] = cg_x[idx] + alpha * cg_p[idx];
    cg_r[idx] = cg_r[idx] - alpha * cg_Ap[idx];
}

// CG update step 2: p = r + beta * p (beta is a scalar on GPU)
@group(0) @binding(0) var<storage, read> cg_r2: array<f64>;
@group(0) @binding(1) var<storage, read_write> cg_p2: array<f64>;
@group(0) @binding(2) var<storage, read> cg_beta: array<f64>;  // beta[0]
@group(0) @binding(3) var<uniform> cg_params2: CGParams;

@compute @workgroup_size(256)
fn cg_update_p(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cg_params2.n) {
        return;
    }

    let beta = cg_beta[0];
    cg_p2[idx] = cg_r2[idx] + beta * cg_p2[idx];
}

// Compute alpha = rz / pAp from two scalar buffers
// Stores result in alpha_out[0]
@group(0) @binding(0) var<storage, read> rz_in: array<f64>;
@group(0) @binding(1) var<storage, read> pap_in: array<f64>;
@group(0) @binding(2) var<storage, read_write> alpha_out: array<f64>;

@compute @workgroup_size(1)
fn compute_alpha(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rz = rz_in[0];
    let pap = pap_in[0];

    if (abs(pap) > 1e-30) {
        alpha_out[0] = rz / pap;
    } else {
        alpha_out[0] = 0.0;
    }
}

// Compute beta = rz_new / rz_old from two scalar buffers
// Also copies rz_new to rz_old for next iteration
@group(0) @binding(0) var<storage, read> rz_new_in: array<f64>;
@group(0) @binding(1) var<storage, read_write> rz_old_inout: array<f64>;
@group(0) @binding(2) var<storage, read_write> beta_out: array<f64>;

@compute @workgroup_size(1)
fn compute_beta(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rz_new = rz_new_in[0];
    let rz_old = rz_old_inout[0];

    if (abs(rz_old) > 1e-30) {
        beta_out[0] = rz_new / rz_old;
    } else {
        beta_out[0] = 0.0;
    }

    // Update rz_old for next iteration
    rz_old_inout[0] = rz_new;
}
