// SPDX-License-Identifier: AGPL-3.0-only
// octave_conv2d.wgsl - Octave Convolution 2D (f64 canonical)

struct Params {
    batch_size: u32,
    in_channels_high: u32,
    in_channels_low: u32,
    out_channels_high: u32,
    out_channels_low: u32,
    in_height_high: u32,
    in_width_high: u32,
    in_height_low: u32,
    in_width_low: u32,
    out_height_high: u32,
    out_width_high: u32,
    out_height_low: u32,
    out_width_low: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    path: u32,
}

@group(0) @binding(0) var<storage, read> input_high: array<f64>;
@group(0) @binding(1) var<storage, read> input_low: array<f64>;
@group(0) @binding(2) var<storage, read> weight: array<f64>;
@group(0) @binding(3) var<storage, read> bias: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

fn avg_pool_2x2(x: u32, y: u32, b: u32, c: u32) -> f64 {
    let base_idx = b * params.in_channels_high * params.in_height_high * params.in_width_high +
                   c * params.in_height_high * params.in_width_high;
    
    var sum: f64 = 0.0;
    for (var dy: u32 = 0u; dy < 2u; dy = dy + 1u) {
        for (var dx: u32 = 0u; dx < 2u; dx = dx + 1u) {
            let iy = y * 2u + dy;
            let ix = x * 2u + dx;
            let idx = base_idx + iy * params.in_width_high + ix;
            sum = sum + input_high[idx];
        }
    }
    return sum / 4.0;
}

fn upsample_bilinear(x: u32, y: u32, b: u32, c: u32) -> f64 {
    let xf = f64(x) / 2.0;
    let yf = f64(y) / 2.0;
    let x0 = u32(floor(xf));
    let y0 = u32(floor(yf));
    let x1 = min(x0 + 1u, params.in_width_low - 1u);
    let y1 = min(y0 + 1u, params.in_height_low - 1u);
    
    let base_idx = b * params.in_channels_low * params.in_height_low * params.in_width_low +
                   c * params.in_height_low * params.in_width_low;
    
    let v00 = input_low[base_idx + y0 * params.in_width_low + x0];
    let v01 = input_low[base_idx + y0 * params.in_width_low + x1];
    let v10 = input_low[base_idx + y1 * params.in_width_low + x0];
    let v11 = input_low[base_idx + y1 * params.in_width_low + x1];
    
    let wx = fract(xf);
    let wy = fract(yf);
    
    return (1.0 - wx) * (1.0 - wy) * v00 +
           wx * (1.0 - wy) * v01 +
           (1.0 - wx) * wy * v10 +
           wx * wy * v11;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c_out = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (params.path == 0u) {
        if (c_out >= params.out_channels_high || oh >= params.out_height_high || ow >= params.out_width_high) {
            return;
        }
    } else if (params.path == 1u) {
        if (c_out >= params.out_channels_low || oh >= params.out_height_low || ow >= params.out_width_low) {
            return;
        }
    } else if (params.path == 2u) {
        if (c_out >= params.out_channels_high || oh >= params.out_height_high || ow >= params.out_width_high) {
            return;
        }
    } else {
        if (c_out >= params.out_channels_low || oh >= params.out_height_low || ow >= params.out_width_low) {
            return;
        }
    }
    
    var out_idx: u32;
    if (params.path == 0u || params.path == 2u) {
        out_idx = b * params.out_channels_high * params.out_height_high * params.out_width_high +
            c_out * params.out_height_high * params.out_width_high +
            oh * params.out_width_high + ow;
    } else {
        out_idx = b * params.out_channels_low * params.out_height_low * params.out_width_low +
            c_out * params.out_height_low * params.out_width_low +
            oh * params.out_width_low + ow;
    }

    output[out_idx] = bias[c_out];
}
