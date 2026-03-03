// SGD (Stochastic Gradient Descent) Optimizer - f64 canonical
//
// Update rule: weight = weight - learning_rate * gradient
//
// Optional momentum:
// velocity = momentum * velocity - learning_rate * gradient
// weight = weight + velocity

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> gradients: array<f64>;
@group(0) @binding(2) var<storage, read> velocity_in: array<f64>;
@group(0) @binding(3) var<storage, read_write> weights_out: array<f64>;
@group(0) @binding(4) var<storage, read_write> velocity_out: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

struct Params {
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&weights) {
        return;
    }

    let w = weights[idx];
    var g = gradients[idx];

    if params.weight_decay != 0.0 {
        g = g + params.weight_decay * w;
    }

    var velocity = 0.0;
    if params.momentum != 0.0 {
        velocity = velocity_in[idx];
        velocity = params.momentum * velocity + (1.0 - params.dampening) * g;
        velocity_out[idx] = velocity;
        g = velocity;
    }

    weights_out[idx] = w - params.learning_rate * g;
}
