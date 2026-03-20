// SPDX-License-Identifier: AGPL-3.0-or-later
// pixel_shuffle_f64.wgsl - Pixel Shuffle (Depth to Space) (f64 canonical)
//
// Rearranges elements in a tensor from depth to spatial dimensions
// Used in super-resolution networks (ESPCN, EDSR)
//
// Transform [B, C*r^2, H, W] → [B, C, H*r, W*r]

struct Params {
    batch_size: u32,
    in_channels: u32,   // C * r^2
    out_channels: u32,  // C
    in_height: u32,     // H
    in_width: u32,      // W
    out_height: u32,    // H * r
    out_width: u32,     // W * r
    upscale_factor: u32, // r
}

@group(0) @binding(0) var<storage, read> input: array<f64>;         // [B, C*r^2, H, W]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [B, C, H*r, W*r]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (c >= params.out_channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    let r = params.upscale_factor;
    
    // Calculate input position
    let ih = oh / r;
    let iw = ow / r;
    let offset_h = oh % r;
    let offset_w = ow % r;
    
    // Channel index in input: c * r^2 + offset_h * r + offset_w
    let ic = c * r * r + offset_h * r + offset_w;
    
    let in_idx = b * params.in_channels * params.in_height * params.in_width +
                 ic * params.in_height * params.in_width +
                 ih * params.in_width +
                 iw;
    
    let out_idx = b * params.out_channels * params.out_height * params.out_width +
                  c * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = input[in_idx];
}
