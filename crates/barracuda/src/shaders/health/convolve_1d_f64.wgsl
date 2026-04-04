// SPDX-License-Identifier: AGPL-3.0-or-later
//
// convolve_1d_f64.wgsl — Parallel 1D valid convolution
//
// out[i] = Σⱼ signal[i + j] * kernel[j]  for j in [0, kernel_len)
//
// Output length = signal_len - kernel_len + 1
//
// Bindings: @0 signal[signal_len], @1 kernel[kernel_len],
//           @2 out[out_len], @3 uniform { signal_len, kernel_len }

struct Params {
    signal_len: u32,
    kernel_len: u32,
}

@group(0) @binding(0) var<storage, read> signal: array<f64>;
@group(0) @binding(1) var<storage, read> kernel: array<f64>;
@group(0) @binding(2) var<storage, read_write> out: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let out_len = params.signal_len - params.kernel_len + 1u;
    if i >= out_len { return; }

    var acc: f64 = 0.0;
    for (var j: u32 = 0u; j < params.kernel_len; j = j + 1u) {
        acc += signal[i + j] * kernel[j];
    }
    out[i] = acc;
}
