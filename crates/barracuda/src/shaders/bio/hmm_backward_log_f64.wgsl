// SPDX-License-Identifier: AGPL-3.0-or-later
//
// hmm_backward_log_f64.wgsl — HMM backward algorithm in log-space
//
// Pairs with hmm_forward_f64.wgsl. Process from t=T-1 backward:
// beta[t][i] = logsumexp_j(log_trans[i][j] + log_emit[t+1][j] + beta[t+1][j])
//
// Bindings: @0 log_trans[S*S], @1 log_emit[T*S], @2 out_beta[T*S], @3 uniform{t_steps, n_states}
//
// GPU dispatch: one thread per sequence (or per time step for batched). Sequential over t backward.
//
// Provenance: neuralSpring → ToadStool absorption

enable f64;

struct HmmBackwardParams {
    t_steps:  u32,
    n_states: u32,
}

@group(0) @binding(0) var<storage, read>       log_trans: array<f64>;   // [S*S] row-major
@group(0) @binding(1) var<storage, read>       log_emit:  array<f64>;   // [T*S] row-major
@group(0) @binding(2) var<storage, read_write> out_beta:  array<f64>;   // [T*S]
@group(0) @binding(3) var<uniform>             params:    HmmBackwardParams;

fn log_sum_exp2(a: f64, b: f64) -> f64 {
    if a < -1.0e300 { return b; }
    if b < -1.0e300 { return a; }
    let mx = max(a, b);
    return mx + log_f64(exp_f64(a - mx) + exp_f64(b - mx));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > 0u { return; }

    let S = params.n_states;
    let T = params.t_steps;

    for (var t = T; t > 0u; t = t - 1u) {
        let t_prev = t - 1u;
        let beta_t_idx = t_prev * S;

        for (var i: u32 = 0u; i < S; i = i + 1u) {
            var acc: f64 = -1.0e300;
            for (var j: u32 = 0u; j < S; j = j + 1u) {
                let lt = log_trans[i * S + j];
                let b_next = select(f64(0.0), out_beta[t * S + j], t < T);
                let le = select(f64(0.0), log_emit[t * S + j], t < T);
                acc = log_sum_exp2(acc, lt + le + b_next);
            }
            out_beta[beta_t_idx + i] = acc;
        }
    }
}
