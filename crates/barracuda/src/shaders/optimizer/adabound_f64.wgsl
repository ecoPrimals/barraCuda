// SPDX-License-Identifier: AGPL-3.0-only
// adabound.wgsl - AdaBound Optimizer (f64 canonical)
//
// Adaptive learning rate optimizer with dynamic bound on learning rates
// Smoothly transitions from adaptive methods to SGD
//
// Reference: "Adaptive Gradient Methods with Dynamic Bound of Learning Rate" by Luo et al. (2019)

struct Params {
    size: u32,
    lr: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
    final_lr: f64,    // Final learning rate for SGD convergence
    gamma: f64,       // Rate of convergence to SGD
    step: u32,        // Current step number
}

@group(0) @binding(0) var<storage, read> grad: array<f64>;           // Gradients
@group(0) @binding(1) var<storage, read_write> param: array<f64>;    // Parameters
@group(0) @binding(2) var<storage, read_write> m: array<f64>;        // First moment
@group(0) @binding(3) var<storage, read_write> v: array<f64>;        // Second moment
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    var g = grad[idx];
    let p = param[idx];

    // Weight decay
    if (params.weight_decay != 0.0) {
        g = g + params.weight_decay * p;
    }

    // Update biased first moment estimate
    m[idx] = params.beta1 * m[idx] + (1.0 - params.beta1) * g;

    // Update biased second raw moment estimate
    v[idx] = params.beta2 * v[idx] + (1.0 - params.beta2) * g * g;

    // Compute bias-corrected estimates
    let step_f = f64(params.step);
    let m_hat = m[idx] / (1.0 - pow_f64(params.beta1, step_f));
    let v_hat = v[idx] / (1.0 - pow_f64(params.beta2, step_f));

    // Compute bounds (AdaBound algorithm)
    let lower_bound = params.final_lr * (1.0 - 1.0 / (params.gamma * step_f + 1.0));
    let upper_bound = params.final_lr * (1.0 + 1.0 / (params.gamma * step_f + 1.0));

    // Compute step size with bounds
    let step_size = params.lr / (sqrt_f64(v_hat) + params.epsilon);
    let bounded_lr = clamp(step_size, lower_bound, upper_bound);

    // Update parameters
    param[idx] = p - bounded_lr * m_hat;
}
