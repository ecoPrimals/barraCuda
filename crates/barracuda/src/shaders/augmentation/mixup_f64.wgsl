// mixup.wgsl - Mixup data augmentation
//
// Mixes two training examples and their labels
// Reference: "mixup: Beyond Empirical Risk Minimization" by Zhang et al. (2018)
//
// x_mixed = λ * x_i + (1 - λ) * x_j
// y_mixed = λ * y_i + (1 - λ) * y_j

struct Params {
    batch_size: u32,
    feature_size: u32,
    lambda: f32,     // Mixing coefficient [0, 1]
    mix_idx: u32,    // Index of sample to mix with
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // [batch_size, feature_size]
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // [batch_size, feature_size]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.feature_size;
    
    if (idx >= total) {
        return;
    }
    
    let b = idx / params.feature_size;
    let f = idx % params.feature_size;
    
    // Get mixing index (circular)
    let mix_b = (b + params.mix_idx) % params.batch_size;
    
    let idx1 = b * params.feature_size + f;
    let idx2 = mix_b * params.feature_size + f;
    
    // Mixup: λ * x_i + (1 - λ) * x_j
    let lam = f64(params.lambda);
    output[idx1] = lam * input[idx1] + (1.0 - lam) * input[idx2];
}
