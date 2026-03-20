// SPDX-License-Identifier: AGPL-3.0-or-later
// deformable_conv2d.wgsl - Deformable Convolution 2D (f64 canonical)
//
// Convolution with learnable offsets for sampling positions

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
    dilation: u32,
    deform_groups: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> offset: array<f64>;
@group(0) @binding(2) var<storage, read> weight: array<f64>;
@group(0) @binding(3) var<storage, read> bias: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

fn bilinear_sample(x: f64, y: f64, b: u32, c: u32) -> f64 {
    let x0 = floor(x);
    let x1 = x0 + 1.0;
    let y0 = floor(y);
    let y1 = y0 + 1.0;
    
    if (x0 < 0.0 || x1 >= f64(params.in_width) || y0 < 0.0 || y1 >= f64(params.in_height)) {
        return 0.0;
    }
    
    let ix0 = u32(x0);
    let ix1 = u32(x1);
    let iy0 = u32(y0);
    let iy1 = u32(y1);
    
    let wa = (x1 - x) * (y1 - y);
    let wb = (x - x0) * (y1 - y);
    let wc = (x1 - x) * (y - y0);
    let wd = (x - x0) * (y - y0);
    
    let idx_a = b * params.in_channels * params.in_height * params.in_width +
                c * params.in_height * params.in_width +
                iy0 * params.in_width + ix0;
    
    let idx_b = b * params.in_channels * params.in_height * params.in_width +
                c * params.in_height * params.in_width +
                iy0 * params.in_width + ix1;
    
    let idx_c = b * params.in_channels * params.in_height * params.in_width +
                c * params.in_height * params.in_width +
                iy1 * params.in_width + ix0;
    
    let idx_d = b * params.in_channels * params.in_height * params.in_width +
                c * params.in_height * params.in_width +
                iy1 * params.in_width + ix1;
    
    return wa * input[idx_a] + wb * input[idx_b] + wc * input[idx_c] + wd * input[idx_d];
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c_out = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (c_out >= params.out_channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    var sum: f64 = 0.0;
    
    for (var c_in: u32 = 0u; c_in < params.in_channels; c_in = c_in + 1u) {
        var k_idx: u32 = 0u;
        for (var kh: u32 = 0u; kh < params.kernel_size; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kernel_size; kw = kw + 1u) {
                let ih_base = f64(oh * params.stride) - f64(params.padding) + f64(kh * params.dilation);
                let iw_base = f64(ow * params.stride) - f64(params.padding) + f64(kw * params.dilation);
                
                let offset_idx = b * 2u * params.kernel_size * params.kernel_size * params.out_height * params.out_width +
                                (2u * k_idx) * params.out_height * params.out_width +
                                oh * params.out_width +
                                ow;
                
                let offset_y = offset[offset_idx];
                let offset_x = offset[offset_idx + params.out_height * params.out_width];
                
                let ih = ih_base + offset_y;
                let iw = iw_base + offset_x;
                
                let sampled_val = bilinear_sample(iw, ih, b, c_in);
                
                let w_idx = c_out * params.in_channels * params.kernel_size * params.kernel_size +
                           c_in * params.kernel_size * params.kernel_size +
                           kh * params.kernel_size +
                           kw;
                
                sum = sum + sampled_val * weight[w_idx];
                k_idx = k_idx + 1u;
            }
        }
    }
    
    sum = sum + bias[c_out];
    
    let out_idx = b * params.out_channels * params.out_height * params.out_width +
                  c_out * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = sum;
}
