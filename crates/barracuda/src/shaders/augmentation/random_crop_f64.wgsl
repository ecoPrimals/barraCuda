// SPDX-License-Identifier: AGPL-3.0-only
// random_crop.wgsl - Random crop augmentation
//
// Randomly crops images to specified size
// Crop positions provided as input (generated on CPU)

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // [B, C, H_in, W_in]
@group(0) @binding(1) var<storage, read> crop_positions: array<u32>; // [B, 2] - (top, left) for each image
@group(0) @binding(2) var<storage, read_write> output: array<f64>;   // [B, C, H_out, W_out]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (c >= params.channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    // Get crop position for this batch item
    let crop_top = crop_positions[b * 2u];
    let crop_left = crop_positions[b * 2u + 1u];
    
    // Calculate input position
    let ih = crop_top + oh;
    let iw = crop_left + ow;
    
    // Read from input
    let in_idx = b * params.channels * params.in_height * params.in_width +
                 c * params.in_height * params.in_width +
                 ih * params.in_width +
                 iw;
    
    let out_idx = b * params.channels * params.out_height * params.out_width +
                  c * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = input[in_idx];
}
