// SPDX-License-Identifier: AGPL-3.0-only
// Runge-Kutta Stage Evaluation - Full f64 Precision
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
// - Full f64 precision for stiff problems
//
// Reference: Dormand & Prince (1980), Cash-Karp coefficients

struct RkParams {
    n: u32,           // State dimension
    stage: u32,       // Current stage (0-5 for RK45)
    _pad0: u32,
    _pad1: u32,
    h: f64,           // Step size
}

@group(0) @binding(0) var<uniform> params: RkParams;
@group(0) @binding(1) var<storage, read> y: array<f64>;           // Current state (n)
@group(0) @binding(2) var<storage, read> k_stages: array<f64>;    // All k stages [6 × n]
@group(0) @binding(3) var<storage, read_write> y_stage: array<f64>;  // y for stage evaluation (n)

// Dormand-Prince a_ij coefficients (row-major, lower triangular)
// Full f64 precision for coefficients
fn get_a(stage: u32, j: u32) -> f64 {
    // Dormand-Prince coefficients — exact rational values
    if (stage == 1u) {
        if (j == 0u) { return f64(1.0) / f64(5.0); }  // 1/5 = 0.2
    } else if (stage == 2u) {
        if (j == 0u) { return f64(3.0) / f64(40.0); }   // 3/40 = 0.075
        if (j == 1u) { return f64(9.0) / f64(40.0); }   // 9/40 = 0.225
    } else if (stage == 3u) {
        if (j == 0u) { return f64(44.0) / f64(45.0); }   // 44/45
        if (j == 1u) { return f64(-56.0) / f64(15.0); }  // -56/15
        if (j == 2u) { return f64(32.0) / f64(9.0); }    // 32/9
    } else if (stage == 4u) {
        if (j == 0u) { return f64(19372.0) / f64(6561.0); }   // 19372/6561
        if (j == 1u) { return f64(-25360.0) / f64(2187.0); }  // -25360/2187
        if (j == 2u) { return f64(64448.0) / f64(6561.0); }   // 64448/6561
        if (j == 3u) { return f64(-212.0) / f64(729.0); }     // -212/729
    } else if (stage == 5u) {
        if (j == 0u) { return f64(9017.0) / f64(3168.0); }    // 9017/3168
        if (j == 1u) { return f64(-355.0) / f64(33.0); }      // -355/33
        if (j == 2u) { return f64(46732.0) / f64(5247.0); }   // 46732/5247
        if (j == 3u) { return f64(49.0) / f64(176.0); }       // 49/176
        if (j == 4u) { return f64(-5103.0) / f64(18656.0); }  // -5103/18656
    }
    return f64(0.0);
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
    
    var sum: f64 = f64(0.0);
    
    // Sum over previous stages
    for (var j = 0u; j < stage; j = j + 1u) {
        let a_sj = get_a(stage, j);
        sum = sum + a_sj * k_stages[j * n + i];
    }
    
    y_stage[i] = y[i] + params.h * sum;
}

// 5th-order solution weights (b_i) — exact rational values
fn get_b5(j: u32) -> f64 {
    // Dormand-Prince 5th order weights
    if (j == 0u) { return f64(35.0) / f64(384.0); }
    if (j == 1u) { return f64(0.0); }
    if (j == 2u) { return f64(500.0) / f64(1113.0); }
    if (j == 3u) { return f64(125.0) / f64(192.0); }
    if (j == 4u) { return f64(-2187.0) / f64(6784.0); }
    if (j == 5u) { return f64(11.0) / f64(84.0); }
    return f64(0.0);
}

// 4th-order solution weights (b*_i) for error estimate
fn get_b4(j: u32) -> f64 {
    // Dormand-Prince 4th order weights
    if (j == 0u) { return f64(5179.0) / f64(57600.0); }
    if (j == 1u) { return f64(0.0); }
    if (j == 2u) { return f64(7571.0) / f64(16695.0); }
    if (j == 3u) { return f64(393.0) / f64(640.0); }
    if (j == 4u) { return f64(-92097.0) / f64(339200.0); }
    if (j == 5u) { return f64(187.0) / f64(2100.0); }
    return f64(0.0);
}

// Error weights: e_i = b_i - b*_i
fn get_e(j: u32) -> f64 {
    return get_b5(j) - get_b4(j);
}

// Compute y_new = y + h * Σ b_i * k_i  (5th order solution)
@group(0) @binding(0) var<uniform> update_params: RkParams;
@group(0) @binding(1) var<storage, read> y_old: array<f64>;
@group(0) @binding(2) var<storage, read> k_all: array<f64>;      // [6 × n]
@group(0) @binding(3) var<storage, read_write> y_new: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn update_solution(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = update_params.n;
    
    if (i >= n) {
        return;
    }
    
    var sum: f64 = f64(0.0);
    for (var j = 0u; j < 6u; j = j + 1u) {
        sum = sum + get_b5(j) * k_all[j * n + i];
    }
    
    y_new[i] = y_old[i] + update_params.h * sum;
}

// Compute error = h * Σ e_i * k_i
@group(0) @binding(0) var<uniform> error_params: RkParams;
@group(0) @binding(1) var<storage, read> k_error: array<f64>;    // [6 × n]
@group(0) @binding(2) var<storage, read_write> error_vec: array<f64>; // [n]

@compute @workgroup_size(256, 1, 1)
fn compute_error(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = error_params.n;
    
    if (i >= n) {
        return;
    }
    
    var sum: f64 = f64(0.0);
    for (var j = 0u; j < 6u; j = j + 1u) {
        sum = sum + get_e(j) * k_error[j * n + i];
    }
    
    error_vec[i] = error_params.h * sum;
}

// Compute scaled error norm: sqrt((1/n) Σ (error_i / scale_i)²)
// where scale_i = atol + rtol * max(|y_i|, |y_new_i|)
struct NormParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    atol: f64,
    rtol: f64,
}

@group(0) @binding(0) var<uniform> norm_params: NormParams;
@group(0) @binding(1) var<storage, read> y_norm: array<f64>;
@group(0) @binding(2) var<storage, read> y_new_norm: array<f64>;
@group(0) @binding(3) var<storage, read> error_in: array<f64>;
@group(0) @binding(4) var<storage, read_write> partial_sums: array<f64>;

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256, 1, 1)
fn error_norm(@builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let i = global_id.x;
    let n = norm_params.n;
    
    var sum_sq: f64 = f64(0.0);
    if (i < n) {
        let y_abs = abs(y_norm[i]);
        let y_new_abs = abs(y_new_norm[i]);
        let scale = norm_params.atol + norm_params.rtol * max(y_abs, y_new_abs);
        let scaled_err = error_in[i] / scale;
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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f64,
    beta: f64,
}

@group(0) @binding(0) var<uniform> axpy_params: AxpyParams;
@group(0) @binding(1) var<storage, read> vec_a: array<f64>;
@group(0) @binding(2) var<storage, read> vec_b: array<f64>;
@group(0) @binding(3) var<storage, read_write> axpy_result: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn axpy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= axpy_params.n) {
        return;
    }
    
    axpy_result[i] = axpy_params.alpha * vec_a[i] + axpy_params.beta * vec_b[i];
}
