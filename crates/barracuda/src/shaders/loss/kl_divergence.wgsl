// KL Divergence (Kullback-Leibler Divergence)
// Measures how one probability distribution differs from another
//
// KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
//
// Key properties:
// - Always non-negative
// - Zero when distributions are identical
// - Asymmetric: KL(P||Q) ≠ KL(Q||P)
// - Not a true distance metric
//
// Used in: VAEs, knowledge distillation, distribution matching

@group(0) @binding(0) var<storage, read> predicted: array<f32>;  // P (predicted distribution)
@group(0) @binding(1) var<storage, read> targets: array<f32>;     // Q (target distribution)
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    size: u32,
    epsilon: f32,  // Small constant to prevent log(0), typically 1e-10
    _padding: vec2<u32>,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= params.size {
        return;
    }
    
    let p = predicted[idx];
    let q = targets[idx];
    
    // Clamp to prevent log(0) and division by zero
    let p_clamped = max(p, params.epsilon);
    let q_clamped = max(q, params.epsilon);
    
    // KL divergence: P * log(P / Q)
    let kl = p_clamped * log(p_clamped / q_clamped);
    
    output[idx] = kl;
}
