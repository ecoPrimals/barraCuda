// SPDX-License-Identifier: AGPL-3.0-or-later
// separable_conv2d.wgsl - Depthwise Separable Convolution 2D (f64 canonical)

struct Params {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    mode: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight: array<f64>;
@group(0) @binding(2) var<storage, read> bias: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    var sum: f64 = 0.0;
    
    if (params.mode == 0u) {
        if (c >= params.in_channels) {
            return;
        }
        
        for (var kh: u32 = 0u; kh < params.kernel_size; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kernel_size; kw = kw + 1u) {
                let ih_raw = i32(oh * params.stride) - i32(params.padding) + i32(kh);
                let iw_raw = i32(ow * params.stride) - i32(params.padding) + i32(kw);
                
                if (ih_raw >= 0 && ih_raw < i32(params.in_height) &&
                    iw_raw >= 0 && iw_raw < i32(params.in_width)) {
                    
                    let ih = u32(ih_raw);
                    let iw = u32(iw_raw);
                    
                    let in_idx = b * params.in_channels * params.in_height * params.in_width +
                                c * params.in_height * params.in_width +
                                ih * params.in_width +
                                iw;
                    
                    let w_idx = c * params.kernel_size * params.kernel_size +
                               kh * params.kernel_size +
                               kw;
                    
                    sum = sum + input[in_idx] * weight[w_idx];
                }
            }
        }
        
        sum = sum + bias[c];
        
    } else {
        if (c >= params.out_channels) {
            return;
        }
        
        for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
            let in_idx = b * params.in_channels * params.out_height * params.out_width +
                        ic * params.out_height * params.out_width +
                        oh * params.out_width +
                        ow;
            
            let w_idx = c * params.in_channels + ic;
            
            sum = sum + input[in_idx] * weight[w_idx];
        }
        
        sum = sum + bias[c];
    }
    
    let out_channels = select(params.in_channels, params.out_channels, params.mode == 1u);
    let out_idx = b * out_channels * params.out_height * params.out_width +
                  c * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = sum;
}
