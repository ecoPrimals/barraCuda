// SPDX-License-Identifier: AGPL-3.0-only
//
// norm_reduce_df64.wgsl — Norm reduction using DF64 workgroup memory
//
// Identical algorithm to norm_reduce_f64.wgsl, but workgroup shared memory
// uses f32-pair (DF64) accumulators instead of native f64. This avoids the
// shared-memory f64 reliability issue on Hybrid-precision devices (NVK,
// Ada Lovelace, consumer GPUs where f64 shared memory returns zeros).
//
// Buffer layout: UNCHANGED — input/output are array<f64>.
//
// Prepend: df64_core.wgsl (auto-injected by Rust wrapper)

struct NormParams {
    size: u32,
    norm_type: u32,
    p: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: NormParams;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn norm_l1_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let v = df64_from_f64(abs(input[gid]));
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
fn norm_l2_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let val = df64_from_f64(input[gid]);
        let sq = df64_mul(val, val);
        shared_hi[tid] = sq.hi;
        shared_lo[tid] = sq.lo;
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
fn norm_linf_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let v = df64_from_f64(abs(input[gid]));
        shared_hi[tid] = v.hi;
        shared_lo[tid] = v.lo;
    } else {
        shared_hi[tid] = 0.0;
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
fn norm_p_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let val = abs(input[gid]);
        let v = df64_from_f64(pow(val, params.p));
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
fn norm_frobenius_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let val = df64_from_f64(input[gid]);
        let sq = df64_mul(val, val);
        shared_hi[tid] = sq.hi;
        shared_lo[tid] = sq.lo;
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
