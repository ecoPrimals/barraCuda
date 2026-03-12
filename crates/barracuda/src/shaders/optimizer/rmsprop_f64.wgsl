// SPDX-License-Identifier: AGPL-3.0-only
// RMSprop Optimizer (f64 canonical)
// Adaptive learning rate method, addresses AdaGrad's diminishing learning rates
//
// Update rule:
// sq_avg = decay * sq_avg + (1 - decay) * gradient²
// weight = weight - learning_rate * gradient / (sqrt(sq_avg) + epsilon)

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> gradients: array<f64>;
@group(0) @binding(2) var<storage, read> sq_avg_in: array<f64>;     // Running average of squared gradients
@group(0) @binding(3) var<storage, read_write> weights_out: array<f64>;
@group(0) @binding(4) var<storage, read_write> sq_avg_out: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

struct Params {
    learning_rate: f64,
    alpha: f64,           // Decay rate, typically 0.99
    epsilon: f64,         // Small constant for numerical stability, typically 1e-8
    weight_decay: f64,    // L2 regularization, typically 0.0
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&weights) {
        return;
    }

    let w = weights[idx];
    var g = gradients[idx];
    let sq = sq_avg_in[idx];

    // Apply weight decay (L2 regularization)
    if params.weight_decay != 0.0 {
        g = g + params.weight_decay * w;
    }

    // Update running average of squared gradients
    let sq_new = params.alpha * sq + (1.0 - params.alpha) * g * g;
    sq_avg_out[idx] = sq_new;

    // Compute adaptive learning rate and update weights
    let adaptive_lr = params.learning_rate / (sqrt_f64(sq_new) + params.epsilon);
    weights_out[idx] = w - adaptive_lr * g;
}
