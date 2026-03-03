// PSNR - Peak Signal-to-Noise Ratio (f64 canonical)
//
// Computes MSE and PSNR for image quality assessment
// PSNR = 10 * log10(MAX^2 / MSE)

struct Params {
    size: u32,
    max_pixel_value: f64,
}

@group(0) @binding(0) var<storage, read> original: array<f64>;
@group(0) @binding(1) var<storage, read> reconstructed: array<f64>;
@group(0) @binding(2) var<storage, read_write> mse: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let diff = original[idx] - reconstructed[idx];
    mse[idx] = diff * diff;
}
