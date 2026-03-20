// SPDX-License-Identifier: AGPL-3.0-or-later
// AvgPool2D - Average pooling compute shader
//
// **Purpose**: Downsampling via local average
// **Input**: [N, C, H, W]
// **Output**: [N, C, H_out, W_out]
//
// **Params**: pool_size (k_h, k_w), stride via uniforms
// Each thread handles one output element

struct AvgPool2DParams {
    n: u32,
    c: u32,
    h_in: u32,
    w_in: u32,
    k_h: u32,
    k_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    h_out: u32,
    w_out: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: AvgPool2DParams;

fn input_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * params.c * params.h_in * params.w_in
         + c * params.h_in * params.w_in
         + h * params.w_in
         + w;
}

fn output_idx(n: u32, c: u32, h: u32, w: u32) -> u32 {
    return n * params.c * params.h_out * params.w_out
         + c * params.h_out * params.w_out
         + h * params.w_out
         + w;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let flat = gid.x;
    let total = params.n * params.c * params.h_out * params.w_out;
    if (flat >= total) { return; }

    let out_w = flat % params.w_out;
    let out_h = (flat / params.w_out) % params.h_out;
    let out_c = (flat / (params.w_out * params.h_out)) % params.c;
    let out_n = flat / (params.w_out * params.h_out * params.c);

    if (out_n >= params.n || out_c >= params.c || out_h >= params.h_out || out_w >= params.w_out) {
        return;
    }

    let in_y_start = i32(out_h) * i32(params.stride_h) - i32(params.pad_h);
    let in_x_start = i32(out_w) * i32(params.stride_w) - i32(params.pad_w);

    var sum = 0.0;
    var count = 0u;

    for (var ky = 0u; ky < params.k_h; ky = ky + 1u) {
        for (var kx = 0u; kx < params.k_w; kx = kx + 1u) {
            let in_h = in_y_start + i32(ky);
            let in_w = in_x_start + i32(kx);

            if (in_h >= 0 && in_h < i32(params.h_in) && in_w >= 0 && in_w < i32(params.w_in)) {
                let idx = input_idx(out_n, out_c, u32(in_h), u32(in_w));
                sum = sum + input[idx];
                count = count + 1u;
            }
        }
    }

    let out_idx = output_idx(out_n, out_c, out_h, out_w);
    output[out_idx] = select(0.0, sum / f32(count), count > 0u);
}
