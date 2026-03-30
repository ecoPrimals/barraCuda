// SPDX-License-Identifier: AGPL-3.0-or-later
//
// sum_reduce_scalar_f64.wgsl — Scalar f64 reduction (no workgroup memory)
//
// Fallback for devices where the DF64 workgroup reduction (shared_hi/shared_lo
// tree reduce with workgroupBarrier) produces incorrect results. Each workgroup
// runs a single thread that sequentially sums its chunk of 256 elements using
// native f64 storage arithmetic. No workgroup memory, no barriers.
//
// Trade-off: 256× fewer threads per workgroup vs DF64 tree reduce, but
// guaranteed correctness on all devices with SHADER_F64 storage support.
//
// Buffer layout: identical to sum_reduce_df64.wgsl — drop-in replacement.

struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

const CHUNK: u32 = 256u;

@compute @workgroup_size(1)
fn sum_reduce_f64(
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let start = wg.x * CHUNK;
    let end = min(start + CHUNK, params.size);
    var acc: f64 = f64(0.0);
    for (var i = start; i < end; i = i + 1u) {
        acc = acc + input[i];
    }
    output[wg.x] = acc;
}

@compute @workgroup_size(1)
fn max_reduce_f64(
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let start = wg.x * CHUNK;
    let end = min(start + CHUNK, params.size);
    var acc: f64 = f64(-3.4028235e+38);
    for (var i = start; i < end; i = i + 1u) {
        if (input[i] > acc) { acc = input[i]; }
    }
    output[wg.x] = acc;
}

@compute @workgroup_size(1)
fn min_reduce_f64(
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let start = wg.x * CHUNK;
    let end = min(start + CHUNK, params.size);
    var acc: f64 = f64(3.4028235e+38);
    for (var i = start; i < end; i = i + 1u) {
        if (input[i] < acc) { acc = input[i]; }
    }
    output[wg.x] = acc;
}
