// SPDX-License-Identifier: AGPL-3.0-or-later
// Conv2D - Full NCHW 2D convolution compute shader
//
// Input:  [N, C_in, H, W]
// Kernel: [C_out, C_in/groups, kH, kW]
// Bias:   [C_out]
// Output: [N, C_out, H_out, W_out]
//
// Supports stride, padding, dilation via uniforms.

struct Conv2DParams {
    n: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    c_out: u32,
    k_h: u32,
    k_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dil_h: u32,
    dil_w: u32,
    h_out: u32,
    w_out: u32,
    groups: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> kernel: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Conv2DParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.n * params.c_out * params.h_out * params.w_out;
    if (flat >= total) { return; }

    let out_w = flat % params.w_out;
    let out_h = (flat / params.w_out) % params.h_out;
    let c_out = (flat / (params.w_out * params.h_out)) % params.c_out;
    let batch = flat / (params.w_out * params.h_out * params.c_out);

    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let group = c_out / c_out_per_group;
    let c_in_start = group * c_in_per_group;

    var sum = 0.0;
    for (var ci = 0u; ci < c_in_per_group; ci = ci + 1u) {
        let c_in_idx = c_in_start + ci;
        for (var ky = 0u; ky < params.k_h; ky = ky + 1u) {
            for (var kx = 0u; kx < params.k_w; kx = kx + 1u) {
                let in_h = i32(out_h) * i32(params.stride_h) + i32(ky) * i32(params.dil_h) - i32(params.pad_h);
                let in_w = i32(out_w) * i32(params.stride_w) + i32(kx) * i32(params.dil_w) - i32(params.pad_w);

                if (in_h >= 0 && in_h < i32(params.h_in) && in_w >= 0 && in_w < i32(params.w_in)) {
                    let i_idx = batch * params.c_in * params.h_in * params.w_in
                              + c_in_idx * params.h_in * params.w_in
                              + u32(in_h) * params.w_in
                              + u32(in_w);
                    let k_idx = c_out * c_in_per_group * params.k_h * params.k_w
                              + ci * params.k_h * params.k_w
                              + ky * params.k_w
                              + kx;
                    sum = sum + input[i_idx] * kernel[k_idx];
                }
            }
        }
    }

    let o_idx = batch * params.c_out * params.h_out * params.w_out
              + c_out * params.h_out * params.w_out
              + out_h * params.w_out
              + out_w;
    output[o_idx] = sum + bias[c_out];
}
