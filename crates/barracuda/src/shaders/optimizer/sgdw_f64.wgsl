// SPDX-License-Identifier: AGPL-3.0-only
// SGDW - SGD with Decoupled Weight Decay (f64 canonical)
// More principled weight decay than L2 regularization
//
// Algorithm:
// 1. Momentum update: v_t = momentum * v_{t-1} + g_t
// 2. Parameter update: w_t = w - lr * v_t - lr * λ * w

struct Params {
    size: u32,
    learning_rate: f64,
    momentum: f64,
    weight_decay: f64,
    dampening: f64,
    nesterov: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> parameters: array<f64>;
@group(0) @binding(2) var<storage, read> gradients: array<f64>;
@group(0) @binding(3) var<storage, read_write> velocity: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let grad = gradients[idx];
    let param = parameters[idx];
    let v = velocity[idx];

    var new_v: f64;
    if (params.momentum > 0.0) {
        if (v == 0.0) {
            new_v = grad;
        } else {
            new_v = params.momentum * v + (1.0 - params.dampening) * grad;
        }
    } else {
        new_v = grad;
    }
    velocity[idx] = new_v;

    var update: f64;
    if (params.nesterov != 0u) {
        update = grad + params.momentum * new_v;
    } else {
        update = new_v;
    }

    let grad_update = params.learning_rate * update;
    let decay_update = params.learning_rate * params.weight_decay * param;

    output[idx] = param - grad_update - decay_update;
}
