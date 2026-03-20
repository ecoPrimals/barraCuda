// SPDX-License-Identifier: AGPL-3.0-or-later
// Grouped Conv2D - Convolution with channel groups (f64 canonical)

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
    groups: u32,
    in_per_group: u32,
    out_per_group: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> kernel: array<f64>;
@group(0) @binding(3) var<storage, read> bias: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z;
    let oc = global_id.y;
    let out_idx = global_id.x;
    
    if (b >= params.batch_size || oc >= params.out_channels || out_idx >= params.out_height * params.out_width) {
        return;
    }
    
    let oh = out_idx / params.out_width;
    let ow = out_idx % params.out_width;
    
    let g = oc / params.out_per_group;
    let oc_local = oc % params.out_per_group;
    
    var sum: f64 = bias[oc];
    
    for (var ic_local = 0u; ic_local < params.in_per_group; ic_local = ic_local + 1u) {
        let ic = g * params.in_per_group + ic_local;
        
        for (var kh = 0u; kh < params.kernel_size; kh = kh + 1u) {
            for (var kw = 0u; kw < params.kernel_size; kw = kw + 1u) {
                let ih_raw = oh * params.stride + kh;
                let iw_raw = ow * params.stride + kw;
                
                if (ih_raw >= params.padding && ih_raw < params.in_height + params.padding &&
                    iw_raw >= params.padding && iw_raw < params.in_width + params.padding) {
                    
                    let ih = ih_raw - params.padding;
                    let iw = iw_raw - params.padding;
                    
                    if (ih < params.in_height && iw < params.in_width) {
                        let in_idx = ((b * params.in_channels + ic) * params.in_height + ih) * params.in_width + iw;
                        let k_idx = ((oc_local * params.in_per_group + ic_local) * params.kernel_size + kh) * params.kernel_size + kw;
                        sum += input[in_idx] * kernel[k_idx];
                    }
                }
            }
        }
    }
    
    let out_idx_flat = ((b * params.out_channels + oc) * params.out_height + oh) * params.out_width + ow;
    output[out_idx_flat] = sum;
}
