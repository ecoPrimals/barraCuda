// SPDX-License-Identifier: AGPL-3.0-or-later
// Parallel mean reduction at DF64 precision (f32-pair).
// Same structure as sum_reduce_df64 but divides by n at the end.
// Requires: df64_core.wgsl (auto-injected by compile_shader_df64)

@group(0) @binding(0) var<storage, read> input: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<f32>>;
@group(0) @binding(2) var<uniform> n: u32;

var<workgroup> shared_hi: array<f32, 256>;
var<workgroup> shared_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;
    let gid = wid.x * 256u + tid;

    if gid < n {
        shared_hi[tid] = input[gid].x;
        shared_lo[tid] = input[gid].y;
    } else {
        shared_hi[tid] = 0.0;
        shared_lo[tid] = 0.0;
    }
    workgroupBarrier();

    var stride = 128u;
    while stride > 0u {
        if tid < stride {
            let a = Df64(shared_hi[tid], shared_lo[tid]);
            let b = Df64(shared_hi[tid + stride], shared_lo[tid + stride]);
            let r = df64_add(a, b);
            shared_hi[tid] = r.hi;
            shared_lo[tid] = r.lo;
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        let sum = Df64(shared_hi[0], shared_lo[0]);
        let r = df64_div(sum, df64_from_f32(f32(n)));
        output[wid.x] = vec2<f32>(r.hi, r.lo);
    }
}
