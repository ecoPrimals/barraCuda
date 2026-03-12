// SPDX-License-Identifier: AGPL-3.0-only
// NAdam Optimizer (Nesterov-accelerated Adaptive Moment Estimation) (f64 canonical)
// Combines Adam with Nesterov momentum for faster convergence
//
// Update rule:
// m = beta1 * m + (1 - beta1) * gradient
// v = beta2 * v + (1 - beta2) * gradient²
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// gradient_nesterov = (beta1 * m_hat + (1 - beta1) * gradient) / (1 - beta1^t)
// weight = weight - learning_rate * gradient_nesterov / (sqrt(v_hat) + epsilon)
//
// Used in: Training deep networks faster than Adam, modern optimizers

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> gradients: array<f64>;
@group(0) @binding(2) var<storage, read> m_in: array<f64>;
@group(0) @binding(3) var<storage, read> v_in: array<f64>;
@group(0) @binding(4) var<storage, read_write> weights_out: array<f64>;
@group(0) @binding(5) var<storage, read_write> m_out: array<f64>;
@group(0) @binding(6) var<storage, read_write> v_out: array<f64>;
@group(0) @binding(7) var<uniform> params: Params;

struct Params {
    learning_rate: f64,
    beta1: f64,           // First moment decay, typically 0.9
    beta2: f64,           // Second moment decay, typically 0.999
    epsilon: f64,         // Numerical stability, typically 1e-8
    weight_decay: f64,    // L2 regularization
    step: u32,            // Current step number
    _padding: vec2<u32>,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&weights) {
        return;
    }

    let w = weights[idx];
    var g = gradients[idx];
    let m = m_in[idx];
    let v = v_in[idx];

    // Apply weight decay
    if params.weight_decay != 0.0 {
        g = g + params.weight_decay * w;
    }

    // Update biased first moment estimate
    let m_new = params.beta1 * m + (1.0 - params.beta1) * g;
    m_out[idx] = m_new;

    // Update biased second moment estimate
    let v_new = params.beta2 * v + (1.0 - params.beta2) * g * g;
    v_out[idx] = v_new;

    // Compute bias correction
    let step_f = f64(params.step);
    let beta1_t = pow_f64(params.beta1, step_f);
    let beta2_t = pow_f64(params.beta2, step_f);

    // Bias-corrected estimates
    let m_hat = m_new / (1.0 - beta1_t);
    let v_hat = v_new / (1.0 - beta2_t);

    // Nesterov momentum
    let gradient_nesterov = (params.beta1 * m_hat + (1.0 - params.beta1) * g) / (1.0 - beta1_t);

    // Update weights with Nesterov-accelerated gradient
    weights_out[idx] = w - params.learning_rate * gradient_nesterov / (sqrt_f64(v_hat) + params.epsilon);
}
