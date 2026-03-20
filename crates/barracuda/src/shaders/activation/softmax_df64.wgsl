// SPDX-License-Identifier: AGPL-3.0-or-later
// Softmax in DF64 precision — single workgroup, 3 phases
// softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@group(0) @binding(0) var<storage, read> input_hi: array<f32>;
@group(0) @binding(1) var<storage, read> input_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lo: array<f32>;
@group(0) @binding(4) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let tid = lid.x;
    let idx = gid.x;

    // Phase 1: find max via tree reduction
    if (idx < size) {
        shared_hi[tid] = input_hi[idx];
        shared_lo[tid] = input_lo[idx];
    } else {
        shared_hi[tid] = -3.4e38;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            if (b.hi > a.hi || (b.hi == a.hi && b.lo > a.lo)) {
                shared_hi[tid] = b.hi;
                shared_lo[tid] = b.lo;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let max_val = Df64(shared_hi[0], shared_lo[0]);

    // Phase 2: compute exp(x - max) and sum
    var exp_val = df64_from_f32(0.0);
    if (idx < size) {
        let x = Df64(input_hi[idx], input_lo[idx]);
        exp_val = exp_df64(df64_sub(x, max_val));
    }
    shared_hi[tid] = exp_val.hi;
    shared_lo[tid] = exp_val.lo;
    workgroupBarrier();

    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let s = df64_add(a, b);
            shared_hi[tid] = s.hi;
            shared_lo[tid] = s.lo;
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    let sum_exp = Df64(shared_hi[0], shared_lo[0]);

    // Phase 3: normalize
    if (idx < size) {
        let result = df64_div(exp_val, sum_exp);
        output_hi[idx] = result.hi;
        output_lo[idx] = result.lo;
    }
}
