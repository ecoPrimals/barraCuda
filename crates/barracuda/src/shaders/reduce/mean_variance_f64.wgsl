// SPDX-License-Identifier: AGPL-3.0-or-later
//
// mean_variance_f64.wgsl — Fused mean + variance in a single pass (Welford)
//
// Computes both mean and variance without intermediate CPU round-trips.
// Uses Welford's online algorithm with parallel merge for numerical stability.
//
// Output layout: [mean, variance] (2 f64 values)
// For sample variance (ddof=1), pass ddof=1 in params.
//
// Absorbed from Kokkos parallel_reduce patterns:
// - Single dispatch replaces separate mean + deviation passes
// - Grid-stride loop amortizes dispatch overhead for large inputs
// - Workgroup tree reduction via shared memory

enable f64;

struct Params {
    n: u32,
    ddof: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> s_count: array<f64, 256>;
var<workgroup> s_mean: array<f64, 256>;
var<workgroup> s_m2: array<f64, 256>;

fn merge_welford(
    ca: f64, ma: f64, m2a: f64,
    cb: f64, mb: f64, m2b: f64,
) -> vec3<f64> {
    let c = ca + cb;
    if c == 0.0 { return vec3<f64>(0.0, 0.0, 0.0); }
    let delta = mb - ma;
    let mean = ma + delta * cb / c;
    let m2 = m2a + m2b + delta * delta * ca * cb / c;
    return vec3<f64>(c, mean, m2);
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

    // Grid-stride accumulation: each thread processes multiple elements
    var local_count: f64 = 0.0;
    var local_mean: f64 = 0.0;
    var local_m2: f64 = 0.0;

    var idx = gid.x;
    while idx < n {
        let x = input[idx];
        local_count += 1.0;
        let delta = x - local_mean;
        local_mean += delta / local_count;
        let delta2 = x - local_mean;
        local_m2 += delta * delta2;
        idx += total_threads;
    }

    s_count[tid] = local_count;
    s_mean[tid] = local_mean;
    s_m2[tid] = local_m2;
    workgroupBarrier();

    // Tree reduction via Welford merge
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            let merged = merge_welford(
                s_count[tid], s_mean[tid], s_m2[tid],
                s_count[tid + stride], s_mean[tid + stride], s_m2[tid + stride],
            );
            s_count[tid] = merged.x;
            s_mean[tid] = merged.y;
            s_m2[tid] = merged.z;
        }
        workgroupBarrier();
    }

    // Single workgroup writes final result
    if tid == 0u {
        if nwg.x == 1u {
            // Final reduction: output [mean, variance]
            let count = s_count[0];
            let divisor = count - f64(params.ddof);
            let variance = select(s_m2[0] / divisor, 0.0, divisor <= 0.0);
            output[0] = s_mean[0];
            output[1] = variance;
        } else {
            // Partial results: [count, mean, M2] per workgroup
            let base = wid.x * 3u;
            output[base] = s_count[0];
            output[base + 1u] = s_mean[0];
            output[base + 2u] = s_m2[0];
        }
    }
}
