// SPDX-License-Identifier: AGPL-3.0-only
// Runge-Kutta Stage Evaluation - Shader-First Implementation
//
// Parallel computation of RK stages for ODE integration.
// Each thread handles one component of the state vector.
//
// For RK4/RK45 (Dormand-Prince):
//   k_i = f(t + c_i*h, y + h * Σ a_ij * k_j)
//
// The key insight: For a state vector y of dimension N,
// ALL N components of a stage can be computed IN PARALLEL.
//
// Sequential part (CPU):
//   - Stage ordering (k1 before k2, etc.)
//   - Step accept/reject
//   - Adaptive step sizing
//
// Parallel part (this shader):
//   - Linear combination: y + h * Σ a_ij * k_j
//   - Error norm computation
//   - Solution update
//
// This is SHADER-FIRST ODE integration:
// - Same math as CPU RK45
// - Parallel over state dimensions
// - Enables massive parallelism for large systems (N >> 1000)
//
// Reference: Dormand & Prince (1980), Cash-Karp coefficients

struct RkParams {
    n: u32,           // State dimension
    h: f32,           // Step size
    stage: u32,       // Current stage (0-5 for RK45)
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: RkParams;
@group(0) @binding(1) var<storage, read> y: array<f32>;           // Current state (n)
@group(0) @binding(2) var<storage, read> k_stages: array<f32>;    // All k stages [6 × n]
@group(0) @binding(3) var<storage, read_write> y_stage: array<f32>;  // y for stage evaluation (n)

// Dormand-Prince a_ij coefficients (row-major, lower triangular)
// a[0] = {} (k1 uses y directly)
// a[1] = {1/5}
// a[2] = {3/40, 9/40}
// a[3] = {44/45, -56/15, 32/9}
// a[4] = {19372/6561, -25360/2187, 64448/6561, -212/729}
// a[5] = {9017/3168, -355/33, 46732/5247, 49/176, -5103/18656}

fn get_a(stage: u32, j: u32) -> f32 {
    // Dormand-Prince coefficients
    if (stage == 1u) {
        if (j == 0u) { return 0.2; }  // 1/5
    } else if (stage == 2u) {
        if (j == 0u) { return 0.075; }   // 3/40
        if (j == 1u) { return 0.225; }   // 9/40
    } else if (stage == 3u) {
        if (j == 0u) { return 0.977777777777778; }   // 44/45
        if (j == 1u) { return -3.733333333333333; }  // -56/15
        if (j == 2u) { return 3.555555555555556; }   // 32/9
    } else if (stage == 4u) {
        if (j == 0u) { return 2.952598689224206; }   // 19372/6561
        if (j == 1u) { return -11.595793324188385; } // -25360/2187
        if (j == 2u) { return 9.822892851699436; }   // 64448/6561
        if (j == 3u) { return -0.290809327846365; }  // -212/729
    } else if (stage == 5u) {
        if (j == 0u) { return 2.846275252525253; }   // 9017/3168
        if (j == 1u) { return -10.757575757575758; } // -355/33
        if (j == 2u) { return 8.906422717743473; }   // 46732/5247
        if (j == 3u) { return 0.278409090909091; }   // 49/176
        if (j == 4u) { return -0.273531303602058; }  // -5103/18656
    }
    return 0.0;
}

// Compute y_stage = y + h * Σ a[stage,j] * k_j
// This prepares y_stage for f(t + c*h, y_stage) evaluation
@compute @workgroup_size(256, 1, 1)
fn prepare_stage(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n;
    let stage = params.stage;
    
    if (i >= n) {
        return;
    }
    
    var sum: f32 = 0.0;
    
    // Sum over previous stages
    for (var j = 0u; j < stage; j = j + 1u) {
        let a_sj = get_a(stage, j);
        sum = sum + a_sj * k_stages[j * n + i];
    }
    
    y_stage[i] = y[i] + params.h * sum;
}

// 5th-order solution weights (b_i)
fn get_b5(j: u32) -> f32 {
    // Dormand-Prince 5th order weights
    if (j == 0u) { return 35.0 / 384.0; }
    if (j == 1u) { return 0.0; }
    if (j == 2u) { return 500.0 / 1113.0; }
    if (j == 3u) { return 125.0 / 192.0; }
    if (j == 4u) { return -2187.0 / 6784.0; }
    if (j == 5u) { return 11.0 / 84.0; }
    return 0.0;
}

// 4th-order solution weights (b*_i) for error estimate
fn get_b4(j: u32) -> f32 {
    // Dormand-Prince 4th order weights
    if (j == 0u) { return 5179.0 / 57600.0; }
    if (j == 1u) { return 0.0; }
    if (j == 2u) { return 7571.0 / 16695.0; }
    if (j == 3u) { return 393.0 / 640.0; }
    if (j == 4u) { return -92097.0 / 339200.0; }
    if (j == 5u) { return 187.0 / 2100.0; }
    return 0.0;
}

// Error weights: e_i = b_i - b*_i
fn get_e(j: u32) -> f32 {
    return get_b5(j) - get_b4(j);
}

// Compute y_new = y + h * Σ b_i * k_i  (5th order solution)
@group(0) @binding(0) var<uniform> update_params: RkParams;
@group(0) @binding(1) var<storage, read> y_old: array<f32>;
@group(0) @binding(2) var<storage, read> k_all: array<f32>;      // [6 × n]
@group(0) @binding(3) var<storage, read_write> y_new: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn update_solution(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = update_params.n;
    
    if (i >= n) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var j = 0u; j < 6u; j = j + 1u) {
        sum = sum + get_b5(j) * k_all[j * n + i];
    }
    
    y_new[i] = y_old[i] + update_params.h * sum;
}

// Compute error = h * Σ e_i * k_i
@group(0) @binding(0) var<uniform> error_params: RkParams;
@group(0) @binding(1) var<storage, read> k_error: array<f32>;    // [6 × n]
@group(0) @binding(2) var<storage, read_write> error: array<f32>; // [n]

@compute @workgroup_size(256, 1, 1)
fn compute_error(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = error_params.n;
    
    if (i >= n) {
        return;
    }
    
    var sum: f32 = 0.0;
    for (var j = 0u; j < 6u; j = j + 1u) {
        sum = sum + get_e(j) * k_error[j * n + i];
    }
    
    error[i] = error_params.h * sum;
}

// Compute scaled error norm: sqrt((1/n) Σ (error_i / scale_i)²)
// where scale_i = atol + rtol * max(|y_i|, |y_new_i|)
struct NormParams {
    n: u32,
    atol: f32,
    rtol: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> norm_params: NormParams;
@group(0) @binding(1) var<storage, read> y_norm: array<f32>;
@group(0) @binding(2) var<storage, read> y_new_norm: array<f32>;
@group(0) @binding(3) var<storage, read> error_vec: array<f32>;
@group(0) @binding(4) var<storage, read_write> partial_sums: array<f32>;

var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn error_norm(@builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let i = global_id.x;
    let n = norm_params.n;
    
    var sum_sq: f32 = 0.0;
    if (i < n) {
        let y_abs = abs(y_norm[i]);
        let y_new_abs = abs(y_new_norm[i]);
        let scale = norm_params.atol + norm_params.rtol * max(y_abs, y_new_abs);
        let scaled_err = error_vec[i] / scale;
        sum_sq = scaled_err * scaled_err;
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
        partial_sums[wg_id.x] = shared_sum[0];
    }
}

// Linear combination: result = alpha * a + beta * b
// Used for FSAL (First Same As Last) optimization
struct AxpyParams {
    n: u32,
    alpha: f32,
    beta: f32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> axpy_params: AxpyParams;
@group(0) @binding(1) var<storage, read> vec_a: array<f32>;
@group(0) @binding(2) var<storage, read> vec_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256, 1, 1)
fn axpy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= axpy_params.n) {
        return;
    }
    
    result[i] = axpy_params.alpha * vec_a[i] + axpy_params.beta * vec_b[i];
}
