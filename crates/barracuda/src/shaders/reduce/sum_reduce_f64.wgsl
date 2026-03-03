// Sum Reduction (f64) - Compute sum over all elements in double precision
//
// CUDA equivalent: thrust::reduce with sum operation (f64)
// Algorithm: Tree reduction (work-efficient) with shared memory
//
// Use cases:
//   - Energy functional integration (nuclear EOS)
//   - RMS error computation
//   - Convergence checking
//   - Any global f64 sum
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Zero unsafe code
// - Self-contained (no external dependencies)
//
// Notes:
// - Single pass produces one partial sum per workgroup
// - For full reduction, dispatch twice: first pass -> partial sums, second pass -> final result
// - f64 zero via f64(0.0) — Naga handles this correctly in storage context

struct ReduceParams {
    size: u32,      // Number of elements to reduce
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // Partial results (one per workgroup)
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> shared_data: array<f64, 256>;

// Sum reduction: output[wg_id] = sum of input elements in this workgroup's range
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn sum_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory (coalesced access)
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Write partial result (one per workgroup)
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Max reduction: output[wg_id] = max of input elements in this workgroup's range
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn max_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data — use large negative for out-of-bounds (f64 min)
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        // Approximate -infinity for f64
        shared_data[tid] = f64(-1e38) * f64(1e38);
    }
    workgroupBarrier();

    // Tree reduction
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

// Min reduction: output[wg_id] = min of input elements in this workgroup's range
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn min_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        // Approximate +infinity for f64
        shared_data[tid] = f64(1e38) * f64(1e38);
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            if (b < a) {
                shared_data[tid] = b;
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Dot product: output[wg_id] = sum(A[i] * B[i]) for this workgroup's range
// Uses A_batch from binding 0 and B_batch from binding 1 (repurposed as second input)
// NOTE: This uses a separate bind group with two input arrays
@group(1) @binding(0) var<storage, read> dot_a: array<f64>;
@group(1) @binding(1) var<storage, read> dot_b: array<f64>;
@group(1) @binding(2) var<storage, read_write> dot_output: array<f64>;
@group(1) @binding(3) var<uniform> dot_params: ReduceParams;

@compute @workgroup_size(256)
fn dot_product_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < dot_params.size) {
        shared_data[tid] = dot_a[gid] * dot_b[gid];
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
        dot_output[workgroup_id.x] = shared_data[0];
    }
}
