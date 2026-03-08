// SPDX-License-Identifier: AGPL-3.0-or-later
//
// sum_reduce_df64.wgsl — Sum/Max/Min reduction using DF64 workgroup memory
//
// Identical algorithm to sum_reduce_f64.wgsl, but workgroup shared memory
// uses f32-pair (DF64) accumulators instead of native f64. This avoids the
// shared-memory f64 reliability issue on Hybrid-precision devices (NVK,
// Titan V, consumer GPUs where f64 shared memory returns zeros).
//
// Buffer layout: UNCHANGED — input/output are array<f64> (device memory f64
// works fine; only workgroup memory is unreliable on Hybrid devices).
//
// Prepend: df64_core.wgsl (auto-injected by Rust wrapper)

struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let v = df64_from_f64(input[gid]);
        shared_hi[tid] = v.hi;
        shared_lo[tid] = v.lo;
    } else {
        shared_hi[tid] = 0.0;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let s = df64_add(a, b);
            shared_hi[tid] = s.hi;
            shared_lo[tid] = s.lo;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}

@compute @workgroup_size(256)
fn max_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let v = df64_from_f64(input[gid]);
        shared_hi[tid] = v.hi;
        shared_lo[tid] = v.lo;
    } else {
        // Approximate -infinity
        shared_hi[tid] = -3.4028235e+38;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a_f64 = df64_to_f64(Df64(shared_hi[tid], shared_lo[tid]));
            let b_f64 = df64_to_f64(Df64(shared_hi[tid + stride], shared_lo[tid + stride]));
            if (b_f64 > a_f64) {
                shared_hi[tid] = shared_hi[tid + stride];
                shared_lo[tid] = shared_lo[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}

@compute @workgroup_size(256)
fn min_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let v = df64_from_f64(input[gid]);
        shared_hi[tid] = v.hi;
        shared_lo[tid] = v.lo;
    } else {
        // Approximate +infinity
        shared_hi[tid] = 3.4028235e+38;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a_f64 = df64_to_f64(Df64(shared_hi[tid], shared_lo[tid]));
            let b_f64 = df64_to_f64(Df64(shared_hi[tid + stride], shared_lo[tid + stride]));
            if (b_f64 < a_f64) {
                shared_hi[tid] = shared_hi[tid + stride];
                shared_lo[tid] = shared_lo[tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}
