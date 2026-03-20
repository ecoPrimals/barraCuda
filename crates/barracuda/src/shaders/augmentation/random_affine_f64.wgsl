// SPDX-License-Identifier: AGPL-3.0-or-later
// RandomAffine - Random affine transformations
//
// Applies random rotation, translation, scale, and shear

struct Params {
    channels: u32,
    height: u32,
    width: u32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    f: f32,
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
    
    let px = f64(x) - f64(params.cx);
    let py = f64(y) - f64(params.cy);
    
    // Inverse transform
    let det = f64(params.a) * f64(params.e) - f64(params.b) * f64(params.d);
    if (abs(det) > 1e-8) {
        let src_x = (f64(params.e) * (px - f64(params.c)) - f64(params.b) * (py - f64(params.f))) / det + f64(params.cx);
        let src_y = (-f64(params.d) * (px - f64(params.c)) + f64(params.a) * (py - f64(params.f))) / det + f64(params.cy);
        
        // Bilinear interpolation
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
                
                let val = v00 * (1.0 - dx) * (1.0 - dy) +
                         v01 * dx * (1.0 - dy) +
                         v10 * (1.0 - dx) * dy +
                         v11 * dx * dy;
                
                output[idx_base + y * params.width + x] = val;
            }
        }
    }
}
