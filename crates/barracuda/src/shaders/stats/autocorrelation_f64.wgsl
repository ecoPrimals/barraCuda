// SPDX-License-Identifier: AGPL-3.0-only
//
// autocorrelation_f64.wgsl — General 1D autocorrelation, single dispatch
//
// Computes C(lag) = (1/(N-lag)) × Σ_{t=0}^{N-lag-1} x[t] × x[t+lag]
// for all lags 0..max_lag in a single dispatch.
//
// One workgroup per lag value; workgroup-internal tree reduction.
//
// Output: array<f64> of length max_lag, where out[k] = C(k).

enable f64;

struct Params {
    n: u32,
    max_lag: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let lag = wid.x;
    if lag >= params.max_lag { return; }

    let tid = lid.x;
    let n = params.n;
    let valid = n - lag;

    var acc: f64 = 0.0;
    var idx = tid;
    while idx < valid {
        acc += input[idx] * input[idx + lag];
        idx += 256u;
    }
    shared[tid] = acc;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            shared[tid] += shared[tid + stride];
        }
        workgroupBarrier();
    }

    if tid == 0u {
        if valid > 0u {
            output[lag] = shared[0] / f64(valid);
        } else {
            output[lag] = 0.0;
        }
    }
}
