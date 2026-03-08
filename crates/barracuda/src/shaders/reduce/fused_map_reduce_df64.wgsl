// SPDX-License-Identifier: AGPL-3.0-or-later
//
// fused_map_reduce_df64.wgsl — Single-dispatch map + reduce using DF64 workgroup memory
//
// Identical algorithm to fused_map_reduce_f64.wgsl, but workgroup shared memory
// uses f32-pair (DF64) accumulators instead of native f64. This avoids the
// shared-memory f64 reliability issue on Hybrid-precision devices.
//
// Buffer layout: UNCHANGED — input/output are array<f64>.
//
// Prepend: df64_core.wgsl (auto-injected by Rust wrapper)
//
// Map operation enum (params.map_op):
//   0 = IDENTITY, 1 = SHANNON, 2 = SIMPSON, 3 = SQUARE, 4 = ABS, 5 = LOG, 6 = NEGATE
//
// Reduce operation enum (params.reduce_op):
//   0 = SUM, 1 = MAX, 2 = MIN, 3 = PRODUCT (log-domain)

struct Params {
    n: u32,
    n_workgroups: u32,
    total: f64,
    map_op: u32,
    reduce_op: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

fn apply_map(x: f64, total: f64, map_op: u32) -> f64 {
    let zero = x - x;
    let one = zero + 1.0;
    let tiny = zero + 1e-300;

    switch (map_op) {
        case 0u: { return x; }
        case 1u: {
            if (x <= zero) { return zero; }
            let p = x / total;
            if (p <= tiny) { return zero; }
            return -p * log_f64(p);
        }
        case 2u: { let p = x / total; return p * p; }
        case 3u: { return x * x; }
        case 4u: { if (x < zero) { return -x; } return x; }
        case 5u: {
            if (x <= zero) { let big = zero + 1e38; return -big * big; }
            return log_f64(x);
        }
        case 6u: { return -x; }
        default: { return x; }
    }
}

fn apply_reduce_df64(a: Df64, b: Df64, reduce_op: u32) -> Df64 {
    switch (reduce_op) {
        case 0u: { return df64_add(a, b); }
        case 1u: {
            let af = df64_to_f64(a);
            let bf = df64_to_f64(b);
            if (af > bf) { return a; }
            return b;
        }
        case 2u: {
            let af = df64_to_f64(a);
            let bf = df64_to_f64(b);
            if (af < bf) { return a; }
            return b;
        }
        case 3u: { return df64_add(a, b); }
        default: { return df64_add(a, b); }
    }
}

fn reduce_identity_df64(reduce_op: u32) -> Df64 {
    switch (reduce_op) {
        case 0u: { return df64_zero(); }
        case 1u: { return Df64(-3.4028235e+38, 0.0); }
        case 2u: { return Df64(3.4028235e+38, 0.0); }
        case 3u: { return df64_zero(); }
        default: { return df64_zero(); }
    }
}

@compute @workgroup_size(256)
fn fused_map_reduce(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = params.n;
    let total = params.total;
    let map_op = params.map_op;
    let reduce_op = params.reduce_op;

    var acc = reduce_identity_df64(reduce_op);
    let grid_stride = params.n_workgroups * 256u;

    var idx = gid;
    while (idx < n) {
        let val = input[idx];
        let mapped = apply_map(val, total, map_op);
        let mapped_df = df64_from_f64(mapped);
        acc = apply_reduce_df64(acc, mapped_df, reduce_op);
        idx = idx + grid_stride;
    }

    shared_hi[tid] = acc.hi;
    shared_lo[tid] = acc.lo;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let r = apply_reduce_df64(a, b, reduce_op);
            shared_hi[tid] = r.hi;
            shared_lo[tid] = r.lo;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        output[workgroup_id.x] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}

@compute @workgroup_size(256)
fn reduce_partials(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let tid = local_id.x;
    let n = params.n;
    let reduce_op = params.reduce_op;

    var val: Df64;
    if (tid < n) {
        val = df64_from_f64(input[tid]);
    } else {
        val = reduce_identity_df64(reduce_op);
    }

    shared_hi[tid] = val.hi;
    shared_lo[tid] = val.lo;
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let r = apply_reduce_df64(a, b, reduce_op);
            shared_hi[tid] = r.hi;
            shared_lo[tid] = r.lo;
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        output[0] = df64_to_f64(Df64(shared_hi[0], shared_lo[0]));
    }
}
