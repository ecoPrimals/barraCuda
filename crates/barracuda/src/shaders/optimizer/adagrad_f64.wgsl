// SPDX-License-Identifier: AGPL-3.0-or-later
// AdaGrad Optimizer (Adaptive Gradient Algorithm) (f64 canonical)
// Adapts learning rate for each parameter based on historical gradients
//
// Update rule:
// accumulated_sq = accumulated_sq + gradient²
// weight = weight - learning_rate * gradient / (sqrt(accumulated_sq) + epsilon)
//
// Benefits: Automatically adapts learning rates for sparse features
// Used in: NLP, sparse data, Google's early deep learning systems

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> gradients: array<f64>;
@group(0) @binding(2) var<storage, read> accumulated_in: array<f64>;  // Sum of squared gradients
@group(0) @binding(3) var<storage, read_write> weights_out: array<f64>;
@group(0) @binding(4) var<storage, read_write> accumulated_out: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

struct Params {
    learning_rate: f64,
    epsilon: f64,         // Numerical stability, typically 1e-8
    weight_decay: f64,    // L2 regularization
    _padding: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&weights) {
        return;
    }

    let w = weights[idx];
    var g = gradients[idx];
    let acc = accumulated_in[idx];

    // Apply weight decay (L2 regularization)
    if params.weight_decay != 0.0 {
        g = g + params.weight_decay * w;
    }

    // Accumulate squared gradients
    let acc_new = acc + g * g;
    accumulated_out[idx] = acc_new;

    // Compute adaptive learning rate and update weights
    let adaptive_lr = params.learning_rate / (sqrt_f64(acc_new) + params.epsilon);
    weights_out[idx] = w - adaptive_lr * g;
}
