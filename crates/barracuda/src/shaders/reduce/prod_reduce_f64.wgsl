// SPDX-License-Identifier: AGPL-3.0-only
// Product Reduction (f64) - Compute product over all elements in double precision
//
// CUDA equivalent: thrust::reduce with multiply operation (f64)
// Algorithm: Tree reduction (work-efficient) with shared memory
//
// Use cases:
//   - Determinant computation
//   - Probability product chains
//   - Partition function terms
//   - Any global f64 product
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Zero unsafe code
// - Self-contained (no external dependencies)
//
// Notes:
// - Identity element for product is 1.0
// - Single pass produces one partial product per workgroup
// - For full reduction, dispatch twice: first pass -> partial products, second pass -> final result

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

// Product reduction: output[wg_id] = product of input elements in this workgroup's range
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn prod_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data into shared memory (coalesced access)
    // Identity element for product is 1.0
    if (gid < params.size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = f64(1.0);
    }
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] * shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Write partial result (one per workgroup)
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Log-product reduction: sum(log(x)) - numerically stable product via log domain
// Useful for very large products that would overflow
// Final product = exp(sum(log(x)))
@compute @workgroup_size(256)
fn log_prod_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load log of data
    if (gid < params.size) {
        let val = input[gid];
        // Use log for positive values, handle zeros specially
        if (val > f64(0.0)) {
            shared_data[tid] = log(val);
        } else {
            // log(0) = -inf, but we use a large negative instead
            shared_data[tid] = f64(-1e38) * f64(1e38);
        }
    } else {
        // Identity: log(1) = 0
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Sum reduction of logs
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Write partial result (sum of logs)
    // Caller takes exp() of final sum to get product
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
