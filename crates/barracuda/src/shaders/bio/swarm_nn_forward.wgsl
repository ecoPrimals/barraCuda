// swarm_nn_forward.wgsl — Population-parallel MLP forward pass (f32)
//
// neuralSpring absorption: neuroevolution swarm controller inference.
//
// Each thread evaluates one (controller, evaluation) pair through a
// configurable 2-layer MLP: input → hidden (sigmoid) → output (argmax).
//
// Weight layout per controller (flat):
//   w1: [input_dim × hidden_dim]  — input-to-hidden
//   b1: [hidden_dim]              — hidden biases
//   w2: [hidden_dim × output_dim] — hidden-to-output
//   b2: [output_dim]              — output biases
//   Total: input_dim*hidden_dim + hidden_dim + hidden_dim*output_dim + output_dim
//
// Bindings:
//   0: weights [n_controllers × weights_per_ctrl] f32
//   1: inputs  [n_controllers × n_evals × input_dim] f32
//   2: actions [n_controllers × n_evals] u32 — argmax output
//   3: params  uniform

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

@group(0) @binding(0) var<storage, read>       weights: array<f32>;
@group(0) @binding(1) var<storage, read>       inputs:  array<f32>;
@group(0) @binding(2) var<storage, read_write> actions: array<u32>;
@group(0) @binding(3) var<uniform>             params:  SwarmParams;

fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-clamp(x, -88.0, 88.0)));
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

    // Weight offsets for this controller
    let w_per = id * hd + hd + hd * od + od;
    let w_base = ctrl * w_per;
    let w1_off = w_base;
    let b1_off = w_base + id * hd;
    let w2_off = b1_off + hd;
    let b2_off = w2_off + hd * od;

    // Input offset
    let in_off = (ctrl * params.n_evals + eval) * id;

    // Hidden layer: sigmoid(W1 @ x + b1)
    // Use registers since hidden_dim is typically small (4-16)
    for (var h = 0u; h < hd; h = h + 1u) {
        var acc = weights[b1_off + h];
        for (var i = 0u; i < id; i = i + 1u) {
            acc = acc + weights[w1_off + i * hd + h] * inputs[in_off + i];
        }
        // Store hidden activation temporarily in first pass
        // (Recomputed inline for output since workgroup memory isn't needed)
        // We'll accumulate output directly below using a two-pass approach
    }

    // Output layer + argmax (fused for register efficiency)
    var best_val = -1e30;
    var best_idx = 0u;

    for (var o = 0u; o < od; o = o + 1u) {
        var out_val = weights[b2_off + o];
        for (var h = 0u; h < hd; h = h + 1u) {
            // Recompute hidden activation for this h
            var hidden = weights[b1_off + h];
            for (var i = 0u; i < id; i = i + 1u) {
                hidden = hidden + weights[w1_off + i * hd + h] * inputs[in_off + i];
            }
            hidden = sigmoid(hidden);
            out_val = out_val + weights[w2_off + h * od + o] * hidden;
        }
        if (out_val > best_val) {
            best_val = out_val;
            best_idx = o;
        }
    }

    actions[idx] = best_idx;
}
