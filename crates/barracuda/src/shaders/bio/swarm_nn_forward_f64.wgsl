// SPDX-License-Identifier: AGPL-3.0-only
//
// swarm_nn_forward_f64.wgsl — Population-parallel MLP forward pass (f64)
//
// Each thread evaluates one (controller, evaluation) pair through a
// 2-layer MLP: input → hidden (sigmoid) → output (argmax).
//
// Weight layout per controller (flat):
//   w1: [input_dim × hidden_dim], b1: [hidden_dim],
//   w2: [hidden_dim × output_dim], b2: [output_dim]
//
// Evolved from f32 → f64 for universal math library portability.

struct SwarmParams {
    n_controllers: u32,
    n_evals:       u32,
    input_dim:     u32,
    hidden_dim:    u32,
    output_dim:    u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
}

@group(0) @binding(0) var<storage, read>       weights: array<f64>;
@group(0) @binding(1) var<storage, read>       inputs:  array<f64>;
@group(0) @binding(2) var<storage, read_write> actions: array<u32>;
@group(0) @binding(3) var<uniform>             params:  SwarmParams;

fn sigmoid_f64(x: f64) -> f64 {
    let clamped = clamp(x, f64(-500.0), f64(500.0));
    return f64(1.0) / (f64(1.0) + exp(-clamped));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.n_controllers * params.n_evals;
    if (idx >= total) { return; }

    let ctrl = idx / params.n_evals;
    let eval = idx % params.n_evals;

    let id = params.input_dim;
    let hd = params.hidden_dim;
    let od = params.output_dim;

    let w_per = id * hd + hd + hd * od + od;
    let w_base = ctrl * w_per;
    let w1_off = w_base;
    let b1_off = w_base + id * hd;
    let w2_off = b1_off + hd;
    let b2_off = w2_off + hd * od;

    let in_off = (ctrl * params.n_evals + eval) * id;

    // Output layer + argmax (fused, recomputing hidden for register efficiency)
    var best_val: f64 = f64(-1e300);
    var best_idx = 0u;

    for (var o = 0u; o < od; o = o + 1u) {
        var out_val = weights[b2_off + o];
        for (var h = 0u; h < hd; h = h + 1u) {
            var hidden = weights[b1_off + h];
            for (var i = 0u; i < id; i = i + 1u) {
                hidden = hidden + weights[w1_off + i * hd + h] * inputs[in_off + i];
            }
            hidden = sigmoid_f64(hidden);
            out_val = out_val + weights[w2_off + h * od + o] * hidden;
        }
        if (out_val > best_val) {
            best_val = out_val;
            best_idx = o;
        }
    }

    actions[idx] = best_idx;
}
