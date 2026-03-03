// elastic_transform.wgsl - Elastic deformation for data augmentation
//
// Elastic deformations: Random displacement fields for image augmentation
// Widely used in medical imaging and handwriting recognition
//
// Simplified implementation using grid-based displacement

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    alpha: f32,      // Scaling of random displacement
    sigma: f32,      // Smoothness of deformation
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;     // [B, C, H, W]
@group(0) @binding(1) var<storage, read> displacement_x: array<f64>; // Random displacement field X
@group(0) @binding(2) var<storage, read> displacement_y: array<f64>; // Random displacement field Y
@group(0) @binding(3) var<storage, read_write> output: array<f64>;   // Deformed output
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let y = global_id.y;
    let x = global_id.x;
    
    if (c >= params.channels || y >= params.height || x >= params.width) {
        return;
    }
    
    // Get displacement for this position
    let disp_idx = y * params.width + x;
    let dx = displacement_x[disp_idx] * f64(params.alpha);
    let dy = displacement_y[disp_idx] * f64(params.alpha);
    
    // Calculate source position
    let src_x = f64(x) + dx;
    let src_y = f64(y) + dy;
    
    // Bilinear interpolation (simplified - nearest neighbor for now)
    let src_x_int = i32(round(src_x));
    let src_y_int = i32(round(src_y));
    
    let out_idx = b * params.channels * params.height * params.width +
                  c * params.height * params.width +
                  y * params.width +
                  x;
    
    // Bounds check
    if (src_x_int >= 0 && src_x_int < i32(params.width) &&
        src_y_int >= 0 && src_y_int < i32(params.height)) {
        
        let in_idx = b * params.channels * params.height * params.width +
                     c * params.height * params.width +
                     u32(src_y_int) * params.width +
                     u32(src_x_int);
        
        output[out_idx] = input[in_idx];
    } else {
        // Out of bounds: fill with 0
        output[out_idx] = 0.0;
    }
}
