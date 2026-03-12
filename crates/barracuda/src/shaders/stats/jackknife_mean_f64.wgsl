// SPDX-License-Identifier: AGPL-3.0-only
// Leave-one-out jackknife for the mean — GPU parallel.
// Thread i computes leave-out-i mean = (full_sum - data[i]) / (n-1).
// Also computes partial (θ_i - θ_bar)^2 for variance reduction on CPU.

struct Params {
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> data: array<f64>;
@group(0) @binding(1) var<storage, read_write> leave_means: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> full_sum_buf: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }

    let full_sum = full_sum_buf[0];
    let n_f = f64(params.n);
    let leave_mean = (full_sum - data[idx]) / (n_f - 1.0);
    leave_means[idx] = leave_mean;
}
