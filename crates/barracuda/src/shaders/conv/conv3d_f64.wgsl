// SPDX-License-Identifier: AGPL-3.0-or-later
// Conv3D - 3D Convolution (f64 canonical)
// Used for video analysis, medical imaging (CT/MRI), spatiotemporal feature extraction

struct Conv3DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_d: u32,
    input_h: u32,
    input_w: u32,
    output_d: u32,
    output_h: u32,
    output_w: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    padding_d: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weights: array<f64>;
@group(0) @binding(2) var<storage, read> bias: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Conv3DParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_d = global_id.z;
    let out_h = global_id.y;
    let out_w = global_id.x;
    
    if (out_d >= params.output_d || out_h >= params.output_h || out_w >= params.output_w) {
        return;
    }
    
    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var out_c = 0u; out_c < params.out_channels; out_c = out_c + 1u) {
            var sum: f64 = 0.0;
            
            for (var in_c = 0u; in_c < params.in_channels; in_c = in_c + 1u) {
                for (var kd = 0u; kd < params.kernel_d; kd = kd + 1u) {
                    for (var kh = 0u; kh < params.kernel_h; kh = kh + 1u) {
                        for (var kw = 0u; kw < params.kernel_w; kw = kw + 1u) {
                            let in_d = i32(out_d * params.stride_d) + i32(kd * params.dilation_d) - i32(params.padding_d);
                            let in_h = i32(out_h * params.stride_h) + i32(kh * params.dilation_h) - i32(params.padding_h);
                            let in_w = i32(out_w * params.stride_w) + i32(kw * params.dilation_w) - i32(params.padding_w);
                            
                            if (in_d >= 0 && in_d < i32(params.input_d) &&
                                in_h >= 0 && in_h < i32(params.input_h) &&
                                in_w >= 0 && in_w < i32(params.input_w)) {
                                
                                let input_idx = b * params.in_channels * params.input_d * params.input_h * params.input_w +
                                              in_c * params.input_d * params.input_h * params.input_w +
                                              u32(in_d) * params.input_h * params.input_w +
                                              u32(in_h) * params.input_w +
                                              u32(in_w);
                                
                                let weight_idx = out_c * params.in_channels * params.kernel_d * params.kernel_h * params.kernel_w +
                                               in_c * params.kernel_d * params.kernel_h * params.kernel_w +
                                               kd * params.kernel_h * params.kernel_w +
                                               kh * params.kernel_w +
                                               kw;
                                
                                sum += input[input_idx] * weights[weight_idx];
                            }
                        }
                    }
                }
            }
            
            sum += bias[out_c];
            
            let output_idx = b * params.out_channels * params.output_d * params.output_h * params.output_w +
                           out_c * params.output_d * params.output_h * params.output_w +
                           out_d * params.output_h * params.output_w +
                           out_h * params.output_w +
                           out_w;
            output[output_idx] = sum;
        }
    }
}
