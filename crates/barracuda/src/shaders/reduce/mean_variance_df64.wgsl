// SPDX-License-Identifier: AGPL-3.0-or-later
//
// mean_variance_df64.wgsl — Fused mean + variance, DF64 core-streaming
//
// Same Welford algorithm as mean_variance_f64.wgsl, but all arithmetic
// runs on the FP32 core array via DF64 (f32-pair, ~48-bit mantissa).
// f64 is used only for buffer I/O (2 ops per element); the O(N) Welford
// accumulation runs entirely in DF64 (~10x throughput on consumer GPUs).
//
// Buffer layout: UNCHANGED from mean_variance_f64.wgsl (array<f64>).
// Prepend: df64_core.wgsl (auto-injected or manual)

struct Params {
    n: u32,
    ddof: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

// Shared memory for Welford triple (count, mean, M2) — split hi/lo
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
    let delta = df64_sub(mb, ma);
    let mean = df64_add(ma, df64_div(df64_mul(delta, cb), c));
    // m2 = m2a + m2b + delta * delta * ca * cb / c
    let d2 = df64_mul(delta, delta);
    let ca_cb = df64_mul(ca, cb);
    let correction = df64_div(df64_mul(d2, ca_cb), c);
    let m2 = df64_add(df64_add(m2a, m2b), correction);
    return array<Df64, 3>(c, mean, m2);
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.n;
    let total_threads = nwg.x * 256u;

    // Grid-stride Welford accumulation in DF64
    var local_count = df64_zero();
    var local_mean = df64_zero();
    var local_m2 = df64_zero();

    var idx = gid.x;
    while idx < n {
        let x = df64_from_f64(input[idx]);
        local_count = df64_add(local_count, df64_from_f32(1.0));
        let delta = df64_sub(x, local_mean);
        local_mean = df64_add(local_mean, df64_div(delta, local_count));
        let delta2 = df64_sub(x, local_mean);
        local_m2 = df64_add(local_m2, df64_mul(delta, delta2));
        idx += total_threads;
    }

    s_count_hi[tid] = local_count.hi; s_count_lo[tid] = local_count.lo;
    s_mean_hi[tid] = local_mean.hi;   s_mean_lo[tid] = local_mean.lo;
    s_m2_hi[tid] = local_m2.hi;       s_m2_lo[tid] = local_m2.lo;
    workgroupBarrier();

    // Tree reduction via Welford merge in DF64
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
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

    if tid == 0u {
        if nwg.x == 1u {
            let count = Df64(s_count_hi[0], s_count_lo[0]);
            let mean = Df64(s_mean_hi[0], s_mean_lo[0]);
            let m2 = Df64(s_m2_hi[0], s_m2_lo[0]);
            let divisor = df64_sub(count, df64_from_f32(f32(params.ddof)));
            var variance = df64_zero();
            // Compare via f64 round-trip: DF64 values can have hi==0 with
            // nonzero lo, so checking only hi misses valid positive divisors.
            if df64_to_f64(divisor) > 0.0 {
                variance = df64_div(m2, divisor);
            }
            output[0] = df64_to_f64(mean);
            output[1] = df64_to_f64(variance);
        } else {
            let base = wid.x * 3u;
            output[base] = df64_to_f64(Df64(s_count_hi[0], s_count_lo[0]));
            output[base + 1u] = df64_to_f64(Df64(s_mean_hi[0], s_mean_lo[0]));
            output[base + 2u] = df64_to_f64(Df64(s_m2_hi[0], s_m2_lo[0]));
        }
    }
}
