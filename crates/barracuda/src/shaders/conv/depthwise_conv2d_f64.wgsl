// SPDX-License-Identifier: AGPL-3.0-only
// Depthwise Conv2D (f64 canonical)
// Efficient convolution that applies a separate filter to each input channel

struct Params {
    batch: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    out_height: u32,
    out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight: array<f64>;
@group(0) @binding(2) var<storage, read> bias: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let batch_channel = global_id.z;
    
    if out_x >= params.out_width || out_y >= params.out_height {
        return;
    }
    
    if batch_channel >= params.batch * params.channels {
        return;
    }
    
    let b = batch_channel / params.channels;
    let c = batch_channel % params.channels;
    
    var sum: f64 = 0.0;
    
    for (var kh: u32 = 0u; kh < params.kernel_h; kh = kh + 1u) {
        for (var kw: u32 = 0u; kw < params.kernel_w; kw = kw + 1u) {
            let in_y_raw = i32(out_y * params.stride_h) + i32(kh) - i32(params.pad_h);
            let in_x_raw = i32(out_x * params.stride_w) + i32(kw) - i32(params.pad_w);
            
            if in_y_raw >= 0 && in_y_raw < i32(params.in_height) &&
               in_x_raw >= 0 && in_x_raw < i32(params.in_width) {
                let in_y = u32(in_y_raw);
                let in_x = u32(in_x_raw);
                
                let input_idx = b * (params.channels * params.in_height * params.in_width) +
                               c * (params.in_height * params.in_width) +
                               in_y * params.in_width +
                               in_x;
                
                let weight_idx = c * (params.kernel_h * params.kernel_w) +
                                kh * params.kernel_w +
                                kw;
                
                sum = sum + input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    sum = sum + bias[c];
    
    let output_idx = b * (params.channels * params.out_height * params.out_width) +
                    c * (params.out_height * params.out_width) +
                    out_y * params.out_width +
                    out_x;
    
    output[output_idx] = sum;
}
