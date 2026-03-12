// SPDX-License-Identifier: AGPL-3.0-only
// Moving window statistics (mean, variance, min, max) (f64 canonical)
// airSpring IoT sensor streams, wetSpring environmental monitoring
//
// Each invocation computes all four statistics for one output position.
// Window slides across the input with stride 1.

struct Params {
    n: u32,          // input length
    window: u32,     // window size
    n_out: u32,      // output length = n - window + 1
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> out_mean: array<f64>;
@group(0) @binding(3) var<storage, read_write> out_var: array<f64>;
@group(0) @binding(4) var<storage, read_write> out_min: array<f64>;
@group(0) @binding(5) var<storage, read_write> out_max: array<f64>;

@compute @workgroup_size(256)
fn moving_window_stats(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_out {
        return;
    }

    let w = params.window;
    var sum: f64 = 0.0;
    var sum_sq: f64 = 0.0;
    var lo: f64 = input[idx];
    var hi: f64 = input[idx];

    for (var j: u32 = 0u; j < w; j = j + 1u) {
        let v = input[idx + j];
        sum = sum + v;
        sum_sq = sum_sq + v * v;
        lo = min(lo, v);
        hi = max(hi, v);
    }

    let mean = sum / f64(w);
    let variance = sum_sq / f64(w) - mean * mean;

    out_mean[idx] = mean;
    out_var[idx] = max(variance, 0.0);
    out_min[idx] = lo;
    out_max[idx] = hi;
}
