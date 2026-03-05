// SPDX-License-Identifier: AGPL-3.0-or-later
//
// correlation_full_f64.wgsl — Fused 5-accumulator Pearson correlation
//
// Computes mean_x, mean_y, var_x, var_y, and pearson_r in a single dispatch.
// No intermediate CPU round-trips between mean and deviation passes.
//
// Output layout: [mean_x, mean_y, var_x, var_y, pearson_r] (5 f64 values)
//
// Absorbed from Kokkos parallel_reduce with JoinOp patterns:
// - Five accumulators fused into one grid-stride + tree reduction
// - Replaces 3+ sequential dispatches with a single kernel launch

enable f64;

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

// 5 shared-memory arrays for the accumulators
var<workgroup> s_sx: array<f64, 256>;
var<workgroup> s_sy: array<f64, 256>;
var<workgroup> s_sxx: array<f64, 256>;
var<workgroup> s_syy: array<f64, 256>;
var<workgroup> s_sxy: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.n;
    let total_threads = nwg.x * 256u;

    // Grid-stride accumulation
    var sx: f64 = 0.0;
    var sy: f64 = 0.0;
    var sxx: f64 = 0.0;
    var syy: f64 = 0.0;
    var sxy: f64 = 0.0;

    var idx = gid.x;
    while idx < n {
        let vx = x[idx];
        let vy = y[idx];
        sx += vx;
        sy += vy;
        sxx += vx * vx;
        syy += vy * vy;
        sxy += vx * vy;
        idx += total_threads;
    }

    s_sx[tid] = sx;
    s_sy[tid] = sy;
    s_sxx[tid] = sxx;
    s_syy[tid] = syy;
    s_sxy[tid] = sxy;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            s_sx[tid] += s_sx[tid + stride];
            s_sy[tid] += s_sy[tid + stride];
            s_sxx[tid] += s_sxx[tid + stride];
            s_syy[tid] += s_syy[tid + stride];
            s_sxy[tid] += s_sxy[tid + stride];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        if nwg.x == 1u {
            let nf = f64(n);
            let mean_x = s_sx[0] / nf;
            let mean_y = s_sy[0] / nf;
            let var_x = s_sxx[0] / nf - mean_x * mean_x;
            let var_y = s_syy[0] / nf - mean_y * mean_y;
            let cov_xy = s_sxy[0] / nf - mean_x * mean_y;

            let denom = sqrt(var_x * var_y);
            let r = select(cov_xy / denom, 0.0, denom < 1.0e-15);

            output[0] = mean_x;
            output[1] = mean_y;
            output[2] = var_x;
            output[3] = var_y;
            output[4] = r;
        } else {
            // Partial sums for multi-workgroup reduction on host
            let base = gid.x / 256u * 5u;
            output[base] = s_sx[0];
            output[base + 1u] = s_sy[0];
            output[base + 2u] = s_sxx[0];
            output[base + 3u] = s_syy[0];
            output[base + 4u] = s_sxy[0];
        }
    }
}
