// SPDX-License-Identifier: AGPL-3.0-or-later
// Max Absolute Difference Reduction (f64) — GPU-Accelerated Convergence Check
//
// Computes: max|a[i] - b[i]| over all elements
//
// Use cases:
//   - SCF convergence: max|E_new - E_old| < tolerance
//   - Iterative solver termination
//   - Energy difference monitoring
//   - Any max-difference convergence criterion
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Zero unsafe code
// - Self-contained (no external dependencies)
//
// Algorithm:
// - Single pass produces one partial max per workgroup
// - Each thread computes |a[i] - b[i]|
// - Tree reduction with max in shared memory
// - For full reduction, dispatch twice if needed

struct DiffParams {
    size: u32,      // Number of elements
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input_a: array<f64>;
@group(0) @binding(1) var<storage, read> input_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // Partial results (one per workgroup)
@group(0) @binding(3) var<uniform> params: DiffParams;

var<workgroup> shared_data: array<f64, 256>;

// Max absolute difference: output[wg_id] = max|a[i] - b[i]| for this workgroup's range
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn max_abs_diff_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load data — compute |a - b| for in-bounds, 0 for out-of-bounds
    if (gid < params.size) {
        let diff = input_a[gid] - input_b[gid];
        shared_data[tid] = abs(diff);
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction with max
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

    // Write partial result (one per workgroup)
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

// Single-array max reduction (for second pass on partial results)
// Uses only input_a for max reduction
@compute @workgroup_size(256)
fn max_reduce_pass2(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load from input_a (used as partial results buffer in pass 2)
    if (gid < params.size) {
        shared_data[tid] = input_a[gid];
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction with max
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
