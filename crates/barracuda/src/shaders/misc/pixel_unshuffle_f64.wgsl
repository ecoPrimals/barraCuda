// SPDX-License-Identifier: AGPL-3.0-only
// pixel_unshuffle_f64.wgsl - Pixel Unshuffle (Space to Depth) (f64 canonical)
//
// Rearranges elements in a tensor from spatial dimensions to depth
// Inverse of pixel shuffle
//
// Transform [B, C, H, W] → [B, C*r^2, H/r, W/r]

struct Params {
    batch_size: u32,
    in_channels: u32,    // C
    out_channels: u32,   // C * r^2
    in_height: u32,      // H
    in_width: u32,       // W
    out_height: u32,     // H / r
    out_width: u32,      // W / r
    downscale_factor: u32, // r
}

@group(0) @binding(0) var<storage, read> input: array<f64>;         // [B, C, H, W]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [B, C*r^2, H/r, W/r]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let oc = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (oc >= params.out_channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    let r = params.downscale_factor;
    
    // Decompose output channel: oc = c * r^2 + offset
    let c = oc / (r * r);
    let offset = oc % (r * r);
    let offset_h = offset / r;
    let offset_w = offset % r;
    
    // Calculate input position
    let ih = oh * r + offset_h;
    let iw = ow * r + offset_w;
    
    let in_idx = b * params.in_channels * params.in_height * params.in_width +
                 c * params.in_height * params.in_width +
                 ih * params.in_width +
                 iw;
    
    let out_idx = b * params.out_channels * params.out_height * params.out_width +
                  oc * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = input[in_idx];
}
