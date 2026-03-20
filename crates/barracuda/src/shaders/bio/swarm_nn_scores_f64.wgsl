// SPDX-License-Identifier: AGPL-3.0-or-later
//
// swarm_nn_scores_f64.wgsl — Max activation output for mean_reduce chaining (f64 canonical)
//
// Same architecture as swarm_nn_forward but outputs max output-layer activation
// (f64) per (controller, eval) for chaining to mean_reduce. Used by
// validate_gpu_pipeline_swarm.
//
// Bindings: 0=params, 1=inputs, 2=scores (RW), 3=config

fn sigmoid(x: f64) -> f64 {
    return f64(1.0) / (f64(1.0) + exp_f64(-x));
}

struct Config {
    n_controllers: u32,
    n_evals: u32,
}

@group(0) @binding(0) var<storage, read> params: array<f64>;
@group(0) @binding(1) var<storage, read> inputs: array<f64>;
@group(0) @binding(2) var<storage, read_write> scores: array<f64>;
@group(0) @binding(3) var<uniform> config: Config;

@compute @workgroup_size(256)
fn swarm_nn_forward_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_controllers = config.n_controllers;
    let n_evals = config.n_evals;
    if idx >= n_controllers * n_evals {
        return;
    }

    let ctrl = idx / n_evals;
    let eval_idx = idx % n_evals;

    let sense = inputs[eval_idx];
    let base = ctrl * 33u;

    var h: array<f64, 4>;
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let w = params[base + i];
        let b = params[base + 4u + i];
        h[i] = sigmoid(sense * w + b);
    }

    var best_val: f64 = f64(-1e9);
    for (var j: u32 = 0u; j < 5u; j = j + 1u) {
        var sum: f64 = params[base + 28u + j];
        for (var i: u32 = 0u; i < 4u; i = i + 1u) {
            sum = sum + h[i] * params[base + 8u + i * 5u + j];
        }
        let o_j = sigmoid(sum);
        if o_j > best_val {
            best_val = o_j;
        }
    }

    scores[idx] = best_val;
}
