// random_rotation.wgsl - Random rotation augmentation (f64 canonical)
//
// Rotates images by random angles
// Rotation matrices provided as input (generated on CPU)

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    fill_value: f64,  // Value for out-of-bounds pixels
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;           // [B, C, H, W]
@group(0) @binding(1) var<storage, read> rotation_matrices: array<f64>; // [B, 4] - [cos, -sin, sin, cos]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;    // [B, C, H, W]
@group(0) @binding(3) var<uniform> params: Params;

// Bilinear interpolation
fn bilinear_sample(x: f64, y: f64, b: u32, c: u32) -> f64 {
    let x0 = floor(x);
    let x1 = x0 + 1.0;
    let y0 = floor(y);
    let y1 = y0 + 1.0;
    
    // Bounds check
    if (x0 < 0.0 || x1 >= f64(params.width) || y0 < 0.0 || y1 >= f64(params.height)) {
        return params.fill_value;
    }
    
    let ix0 = u32(x0);
    let ix1 = u32(x1);
    let iy0 = u32(y0);
    let iy1 = u32(y1);
    
    let wx = x - x0;
    let wy = y - y0;
    
    let base = b * params.channels * params.height * params.width +
               c * params.height * params.width;
    
    let v00 = input[base + iy0 * params.width + ix0];
    let v01 = input[base + iy0 * params.width + ix1];
    let v10 = input[base + iy1 * params.width + ix0];
    let v11 = input[base + iy1 * params.width + ix1];
    
    return (1.0 - wx) * (1.0 - wy) * v00 +
           wx * (1.0 - wy) * v01 +
           (1.0 - wx) * wy * v10 +
           wx * wy * v11;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (c >= params.channels || oh >= params.height || ow >= params.width) {
        return;
    }
    
    // Center coordinates
    let cx = f64(params.width) / 2.0;
    let cy = f64(params.height) / 2.0;
    
    // Output coordinates relative to center
    let ox = f64(ow) - cx;
    let oy = f64(oh) - cy;
    
    // Load rotation matrix
    let cos_theta = rotation_matrices[b * 4u + 0u];
    let neg_sin_theta = rotation_matrices[b * 4u + 1u];
    let sin_theta = rotation_matrices[b * 4u + 2u];
    // cos_theta again at index 3
    
    // Apply inverse rotation to find source coordinates
    let ix = cos_theta * ox + neg_sin_theta * oy + cx;
    let iy = sin_theta * ox + cos_theta * oy + cy;
    
    // Sample with bilinear interpolation
    let value = bilinear_sample(ix, iy, b, c);
    
    let out_idx = b * params.channels * params.height * params.width +
                  c * params.height * params.width +
                  oh * params.width +
                  ow;
    
    output[out_idx] = value;
}
