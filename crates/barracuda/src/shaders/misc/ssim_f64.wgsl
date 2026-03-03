// SSIM - Structural Similarity Index Measure (f64 canonical)
//
// Computes SSIM metric for image quality assessment
// Uses sliding window approach with luminance, contrast, and structure components

struct Params {
    width: u32,
    height: u32,
    window_size: u32,
    c1: f64,
    c2: f64,
    num_windows: u32,
}

@group(0) @binding(0) var<storage, read> image1: array<f64>;
@group(0) @binding(1) var<storage, read> image2: array<f64>;
@group(0) @binding(2) var<storage, read_write> window_ssim: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let window_x = global_id.x;
    let window_y = global_id.y;
    
    let num_windows_x = params.width - params.window_size + 1u;
    let num_windows_y = params.height - params.window_size + 1u;
    
    if (window_x >= num_windows_x || window_y >= num_windows_y) {
        return;
    }
    
    let window_idx = window_y * num_windows_x + window_x;
    
    // Compute statistics in window
    var sum1: f64 = 0.0;
    var sum2: f64 = 0.0;
    var sum1_sq: f64 = 0.0;
    var sum2_sq: f64 = 0.0;
    var sum12: f64 = 0.0;
    let n = f64(params.window_size * params.window_size);
    
    for (var wi = 0u; wi < params.window_size; wi = wi + 1u) {
        for (var wj = 0u; wj < params.window_size; wj = wj + 1u) {
            let px = window_x + wj;
            let py = window_y + wi;
            let idx = py * params.width + px;
            
            let val1 = image1[idx];
            let val2 = image2[idx];
            
            sum1 = sum1 + val1;
            sum2 = sum2 + val2;
            sum1_sq = sum1_sq + val1 * val1;
            sum2_sq = sum2_sq + val2 * val2;
            sum12 = sum12 + val1 * val2;
        }
    }
    
    let mean1 = sum1 / n;
    let mean2 = sum2 / n;
    let var1 = sum1_sq / n - mean1 * mean1;
    let var2 = sum2_sq / n - mean2 * mean2;
    let covar = sum12 / n - mean1 * mean2;
    
    // SSIM formula
    let numerator = (2.0 * mean1 * mean2 + params.c1) * (2.0 * covar + params.c2);
    let denominator = (mean1 * mean1 + mean2 * mean2 + params.c1) * (var1 + var2 + params.c2);
    
    window_ssim[window_idx] = numerator / denominator;
}
