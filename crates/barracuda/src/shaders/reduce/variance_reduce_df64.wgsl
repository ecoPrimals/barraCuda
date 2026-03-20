// SPDX-License-Identifier: AGPL-3.0-or-later
//
// variance_reduce_df64.wgsl — Welford variance reduction using DF64 workgroup memory
//
// Identical algorithm to variance_reduce_f64.wgsl, but workgroup shared memory
// uses f32-pair (DF64) accumulators instead of native f64. This avoids the
// shared-memory f64 reliability issue on Hybrid-precision devices (NVK,
// Titan V, consumer GPUs where f64 shared memory returns zeros).
//
// Buffer layout: UNCHANGED — input/output are array<f64>.
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

var<workgroup> s_count_hi: array<f32, 256>;
var<workgroup> s_count_lo: array<f32, 256>;
var<workgroup> s_mean_hi: array<f32, 256>;
var<workgroup> s_mean_lo: array<f32, 256>;
var<workgroup> s_m2_hi: array<f32, 256>;
var<workgroup> s_m2_lo: array<f32, 256>;

fn merge_welford_df64(
    ca: Df64, ma: Df64, m2a: Df64,
    cb: Df64, mb: Df64, m2b: Df64,
) -> array<Df64, 3> {
    let c = df64_add(ca, cb);
    if c.hi == 0.0 && c.lo == 0.0 {
        return array<Df64, 3>(df64_zero(), df64_zero(), df64_zero());
    }
    // @ilp_region begin — delta, d2, ca_cb are mutually independent
    let delta = df64_sub(mb, ma);
    let d2 = df64_mul(delta, delta);
    let ca_cb = df64_mul(ca, cb);
    let mean = df64_add(ma, df64_div(df64_mul(delta, cb), c));
    let correction = df64_div(df64_mul(d2, ca_cb), c);
    let m2 = df64_add(df64_add(m2a, m2b), correction);
    // @ilp_region end
    return array<Df64, 3>(c, mean, m2);
}

@compute @workgroup_size(256)
fn variance_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.size) {
        let x = df64_from_f64(input[gid]);
        let one = df64_from_f32(1.0);
        s_count_hi[tid] = one.hi; s_count_lo[tid] = one.lo;
        s_mean_hi[tid] = x.hi;   s_mean_lo[tid] = x.lo;
        s_m2_hi[tid] = 0.0;      s_m2_lo[tid] = 0.0;
    } else {
        s_count_hi[tid] = 0.0; s_count_lo[tid] = 0.0;
        s_mean_hi[tid] = 0.0;  s_mean_lo[tid] = 0.0;
        s_m2_hi[tid] = 0.0;    s_m2_lo[tid] = 0.0;
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let merged = merge_welford_df64(
                Df64(s_count_hi[tid], s_count_lo[tid]),
                Df64(s_mean_hi[tid], s_mean_lo[tid]),
                Df64(s_m2_hi[tid], s_m2_lo[tid]),
                Df64(s_count_hi[tid + stride], s_count_lo[tid + stride]),
                Df64(s_mean_hi[tid + stride], s_mean_lo[tid + stride]),
                Df64(s_m2_hi[tid + stride], s_m2_lo[tid + stride]),
            );
            s_count_hi[tid] = merged[0].hi; s_count_lo[tid] = merged[0].lo;
            s_mean_hi[tid] = merged[1].hi;  s_mean_lo[tid] = merged[1].lo;
            s_m2_hi[tid] = merged[2].hi;    s_m2_lo[tid] = merged[2].lo;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let base = workgroup_id.x * 3u;
        output[base] = df64_to_f64(Df64(s_count_hi[0], s_count_lo[0]));
        output[base + 1u] = df64_to_f64(Df64(s_mean_hi[0], s_mean_lo[0]));
        output[base + 2u] = df64_to_f64(Df64(s_m2_hi[0], s_m2_lo[0]));
    }
}
