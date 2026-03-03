// Norm Reduction (f64) - Compute p-norm over all elements in double precision
//
// Formula: ||x||_p = (sum(|x_i|^p))^(1/p)
// Special cases:
//   - p=1: L1 norm (sum of absolute values)
//   - p=2: L2 norm (Euclidean length)
//   - p=inf: Max norm (maximum absolute value)
//
// CUDA equivalent: cuBLAS nrm2 (for L2) or custom kernel
// Algorithm: Tree reduction (work-efficient) with shared memory
//
// Use cases:
//   - Vector norms for convergence checks
//   - Error metrics (L1, L2, Linf)
//   - Regularization terms
//   - Scientific computing
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Zero unsafe code
// - Self-contained (no external dependencies)
//
// Notes:
// - This computes sum(|x|^p), the p-th root is taken on CPU
// - For L2 norm, sum of squares is returned (caller takes sqrt)
// - For inf norm, uses max reduction

struct NormParams {
    size: u32,      // Number of elements
    norm_type: u32, // 1=L1, 2=L2, 0=Linf
    p: f64,         // Generic p-norm parameter (for p != 1,2,inf)
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // Partial results
@group(0) @binding(2) var<uniform> params: NormParams;

var<workgroup> shared_data: array<f64, 256>;

// L1 norm: sum(|x|)
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn norm_l1_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        shared_data[tid] = abs(input[gid]);
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction (sum)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// L2 norm: sqrt(sum(x^2)) - returns sum of squares (sqrt on CPU)
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn norm_l2_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let val = input[gid];
        shared_data[tid] = val * val;
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction (sum of squares)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Linf norm: max(|x|)
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn norm_linf_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        shared_data[tid] = abs(input[gid]);
    } else {
        // Identity for max is negative infinity (approximated)
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction (max)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            if (b > a) {
                shared_data[tid] = b;
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Generic p-norm: (sum(|x|^p))^(1/p) - returns sum of |x|^p (root on CPU)
// For p != 1, 2, inf
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn norm_p_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let val = abs(input[gid]);
        shared_data[tid] = pow(val, params.p);
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction (sum)
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Frobenius norm (for matrices): sqrt(sum(|a_ij|^2))
// Same as L2 but semantically for matrices
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn norm_frobenius_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let val = input[gid];
        shared_data[tid] = val * val;
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
