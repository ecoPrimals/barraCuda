// SPDX-License-Identifier: AGPL-3.0-or-later
// LAMB - Layer-wise Adaptive Moments optimizer for Batch training (f64 canonical)
// Enables large batch training (e.g., BERT with 64K batch size)
// Combines Adam with layer-wise adaptation (trust ratio)
//
// Algorithm:
// 1. Compute Adam update: m_t = β1*m + (1-β1)*g, v_t = β2*v + (1-β2)*g^2
// 2. Bias correction: m_hat = m / (1 - β1^t), v_hat = v / (1 - β2^t)
// 3. Adam step: r_t = m_hat / (sqrt(v_hat) + ε)
// 4. Trust ratio: φ = ||w|| / ||r_t||
// 5. Update: w_t = w - lr * φ * r_t

struct Params {
    size: u32,
    step: u32,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> parameters: array<f64>;
@group(0) @binding(2) var<storage, read> gradients: array<f64>;
@group(0) @binding(3) var<storage, read_write> momentum: array<f64>;
@group(0) @binding(4) var<storage, read_write> variance: array<f64>;
@group(0) @binding(5) var<storage, read_write> adam_step: array<f64>;  // r_t (intermediate)
@group(0) @binding(6) var<storage, read_write> output: array<f64>;

// Step 1: Compute Adam update and store r_t
@compute @workgroup_size(256)
fn compute_adam_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let grad = gradients[idx] + params.weight_decay * parameters[idx];

    // Update momentum and variance
    let m = params.beta1 * momentum[idx] + (1.0 - params.beta1) * grad;
    momentum[idx] = m;

    let v = params.beta2 * variance[idx] + (1.0 - params.beta2) * grad * grad;
    variance[idx] = v;

    // Bias correction
    let step_f = f64(params.step);
    let m_hat = m / (1.0 - pow_f64(params.beta1, step_f));
    let v_hat = v / (1.0 - pow_f64(params.beta2, step_f));

    // Compute Adam step: r_t = m_hat / (sqrt(v_hat) + ε)
    let r_t = m_hat / (sqrt_f64(v_hat) + params.epsilon);
    adam_step[idx] = r_t;
}

// Step 2: Compute trust ratio and apply update
@compute @workgroup_size(1)
fn apply_trust_ratio(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var param_norm_sq: f64 = 0.0;
    var step_norm_sq: f64 = 0.0;

    for (var i = 0u; i < params.size; i = i + 1u) {
        let p = parameters[i];
        let r = adam_step[i];
        param_norm_sq = param_norm_sq + p * p;
        step_norm_sq = step_norm_sq + r * r;
    }

    let param_norm = sqrt_f64(param_norm_sq);
    let step_norm = sqrt_f64(step_norm_sq);

    // Compute trust ratio: φ = ||w|| / ||r_t||
    var trust_ratio: f64 = 1.0;
    if (step_norm > params.epsilon) {
        trust_ratio = param_norm / step_norm;
    }

    // Apply update with trust ratio
    for (var i = 0u; i < params.size; i = i + 1u) {
        let update = params.learning_rate * trust_ratio * adam_step[i];
        output[i] = parameters[i] - update;
    }
}
