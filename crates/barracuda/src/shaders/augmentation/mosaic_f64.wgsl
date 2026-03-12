// SPDX-License-Identifier: AGPL-3.0-only
// Mosaic - Mosaic augmentation (YOLO-style)
//
// Combines 4 images into one mosaic

struct Params {
    channels: u32,
    height: u32,
    width: u32,
    split_x: u32,
    split_y: u32,
}

@group(0) @binding(0) var<storage, read> image0: array<f64>;
@group(0) @binding(1) var<storage, read> image1: array<f64>;
@group(0) @binding(2) var<storage, read> image2: array<f64>;
@group(0) @binding(3) var<storage, read> image3: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    let spatial_idx = y * params.width + x;
    
    // Determine which image to use based on quadrant
    var src_image: u32;
    if (y < params.split_y) {
        if (x < params.split_x) {
            src_image = 0u; // Top-left
        } else {
            src_image = 1u; // Top-right
        }
    } else {
        if (x < params.split_x) {
            src_image = 2u; // Bottom-left
        } else {
            src_image = 3u; // Bottom-right
        }
    }
    
    // Copy pixel from appropriate source image
    for (var c = 0u; c < params.channels; c = c + 1u) {
        let src_idx = c * params.height * params.width + spatial_idx;
        let dst_idx = c * params.height * params.width + spatial_idx;
        
        if (src_image == 0u) {
            output[dst_idx] = image0[src_idx];
        } else if (src_image == 1u) {
            output[dst_idx] = image1[src_idx];
        } else if (src_image == 2u) {
            output[dst_idx] = image2[src_idx];
        } else {
            output[dst_idx] = image3[src_idx];
        }
    }
}
