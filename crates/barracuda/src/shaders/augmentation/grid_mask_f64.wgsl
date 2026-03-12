// SPDX-License-Identifier: AGPL-3.0-only
// GridMask - Grid-based masking augmentation
//
// Masks structured grid regions in images with rotation support

struct Params {
    channels: u32,
    height: u32,
    width: u32,
    ratio: f32,
    rotate: f32, // degrees
    grid_size: u32,
    offset_x: u32,
    offset_y: u32,
    mask_size: u32,
    cos_a: f32,
    sin_a: f32,
    cx: f32,
    cy: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;
    
    if (x >= params.width || y >= params.height) {
        return;
    }
    
    // Rotate coordinates
    let px = f64(x) - f64(params.cx);
    let py = f64(y) - f64(params.cy);
    let rot_x = px * f64(params.cos_a) - py * f64(params.sin_a) + f64(params.cx);
    let rot_y = px * f64(params.sin_a) + py * f64(params.cos_a) + f64(params.cy);
    
    let rot_x_int = i32(rot_x);
    let rot_y_int = i32(rot_y);
    
    if (rot_x_int >= 0 && rot_x_int < i32(params.width) && 
        rot_y_int >= 0 && rot_y_int < i32(params.height)) {
        let grid_x = (u32(rot_x_int) + params.offset_x) / params.grid_size;
        let grid_y = (u32(rot_y_int) + params.offset_y) / params.grid_size;
        
        // Mask alternating grid cells
        if ((grid_x + grid_y) % 2u == 0u) {
            let local_x = (u32(rot_x_int) + params.offset_x) % params.grid_size;
            let local_y = (u32(rot_y_int) + params.offset_y) % params.grid_size;
            
            if (local_x < params.mask_size && local_y < params.mask_size) {
                // Mask this pixel for all channels
                for (var c = 0u; c < params.channels; c = c + 1u) {
                    let idx = c * params.height * params.width + y * params.width + x;
                    output[idx] = 0.0;
                }
                return;
            }
        }
    }
    
    // Copy input to output for unmasked pixels
    for (var c = 0u; c < params.channels; c = c + 1u) {
        let idx = c * params.height * params.width + y * params.width + x;
        output[idx] = input[idx];
    }
}
