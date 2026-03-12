// SPDX-License-Identifier: AGPL-3.0-only
// RandomPerspective - Random perspective transformation (f64 canonical)
//
// Applies random perspective distortion

struct Params {
    channels: u32,
    height: u32,
    width: u32,
    dst_corner0: vec2<f64>,
    dst_corner1: vec2<f64>,
    dst_corner2: vec2<f64>,
    dst_corner3: vec2<f64>,
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
    
    let u = f64(x) / f64(params.width);
    let v = f64(y) / f64(params.height);
    
    // Bilinear interpolation of perspective coordinates
    let top_x = params.dst_corner0.x * (1.0 - u) + params.dst_corner1.x * u;
    let top_y = params.dst_corner0.y * (1.0 - u) + params.dst_corner1.y * u;
    let bottom_x = params.dst_corner3.x * (1.0 - u) + params.dst_corner2.x * u;
    let bottom_y = params.dst_corner3.y * (1.0 - u) + params.dst_corner2.y * u;
    
    let src_x = top_x * (1.0 - v) + bottom_x * v;
    let src_y = top_y * (1.0 - v) + bottom_y * v;
    
    // Sample from source with bilinear interpolation
    if (src_x >= 0.0 && src_x < f64(params.width) - 1.0 &&
        src_y >= 0.0 && src_y < f64(params.height) - 1.0) {
        let x0 = u32(src_x);
        let y0 = u32(src_y);
        let dx = src_x - f64(x0);
        let dy = src_y - f64(y0);
        
        for (var c = 0u; c < params.channels; c = c + 1u) {
            let idx_base = c * params.height * params.width;
            let v00 = input[idx_base + y0 * params.width + x0];
            let v01 = input[idx_base + y0 * params.width + x0 + 1u];
            let v10 = input[idx_base + (y0 + 1u) * params.width + x0];
            let v11 = input[idx_base + (y0 + 1u) * params.width + x0 + 1u];
            
            output[idx_base + y * params.width + x] = v00 * (1.0 - dx) * (1.0 - dy) +
                                                      v01 * dx * (1.0 - dy) +
                                                      v10 * (1.0 - dx) * dy +
                                                      v11 * dx * dy;
        }
    }
}
