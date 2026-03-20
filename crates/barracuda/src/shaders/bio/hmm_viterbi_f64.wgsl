// SPDX-License-Identifier: AGPL-3.0-or-later
//
// hmm_viterbi_f64.wgsl — Viterbi decoding in log-space
//
// delta[t][j] = max_i(delta[t-1][i] + log_trans[i][j]) + log_emit[t][j]
// Track argmax for backtracking. out_path[T] holds the decoded state sequence.
//
// Bindings: @0 log_trans[S*S], @1 log_emit[T*S], @2 log_init[S], @3 out_path[T],
//          @4 out_delta[T*S], @5 out_psi[T*S] (argmax), @6 uniform{t_steps, n_states}
//
// GPU dispatch: one thread for full forward pass, then backtrack (or CPU backtrack).
//
// Provenance: neuralSpring → ToadStool absorption

enable f64;

struct HmmViterbiParams {
    t_steps:  u32,
    n_states: u32,
}

@group(0) @binding(0) var<storage, read>       log_trans: array<f64>;   // [S*S] row-major
@group(0) @binding(1) var<storage, read>       log_emit:  array<f64>;   // [T*S] row-major
@group(0) @binding(2) var<storage, read>       log_init:  array<f64>;   // [S]
@group(0) @binding(3) var<storage, read_write> out_path:  array<u32>;   // [T]
@group(0) @binding(4) var<storage, read_write> out_delta: array<f64>;  // [T*S]
@group(0) @binding(5) var<storage, read_write> out_psi:   array<u32>;   // [T*S] argmax for backtrack
@group(0) @binding(6) var<uniform>             params:    HmmViterbiParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > 0u { return; }

    let S = params.n_states;
    let T = params.t_steps;

    for (var t: u32 = 0u; t < T; t = t + 1u) {
        for (var j: u32 = 0u; j < S; j = j + 1u) {
            if t == 0u {
                out_delta[j] = log_init[j] + log_emit[j];
                out_psi[j] = 0u;
            } else {
                var best_val: f64 = -1.0e300;
                var best_i: u32 = 0u;
                for (var i: u32 = 0u; i < S; i = i + 1u) {
                    let val = out_delta[(t - 1u) * S + i] + log_trans[i * S + j];
                    if val > best_val {
                        best_val = val;
                        best_i = i;
                    }
                }
                out_delta[t * S + j] = best_val + log_emit[t * S + j];
                out_psi[t * S + j] = best_i;
            }
        }
    }

    var best_j: u32 = 0u;
    var best_val: f64 = -1.0e300;
    for (var j: u32 = 0u; j < S; j = j + 1u) {
        let val = out_delta[(T - 1u) * S + j];
        if val > best_val {
            best_val = val;
            best_j = j;
        }
    }

    out_path[T - 1u] = best_j;
    for (var t = T; t > 1u; t = t - 1u) {
        out_path[t - 2u] = out_psi[(t - 1u) * S + out_path[t - 1u]];
    }
}
