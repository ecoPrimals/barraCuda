// SPDX-License-Identifier: AGPL-3.0-only
//
// hmm_forward_f64.wgsl — Full-pass HMM Forward Algorithm (f64)
//
// Computes the forward variable α_t(j) for all states j at timestep t
// in a single dispatch per timestep.  Replaces the per-step Tensor loop
// pattern used in neuralSpring.
//
// α_t(j) = emit(j, O_t) × Σ_i α_{t-1}(i) × trans(i, j)
//
// Log-domain for numerical stability:
//   log α_t(j) = log_emit(j, O_t) + logsumexp_i(log α_{t-1}(i) + log_trans(i, j))
//
// Layout:
//   log_alpha_prev: [S] f64  — log α at previous timestep
//   log_alpha_next: [S] f64  — log α at current timestep (output)
//   log_trans:      [S×S] f64 — log transition matrix (row-major: [from × to])
//   log_emit:       [T×S] f64 — log emission probs, pre-extracted per timestep
//
// Dispatch: (ceil(S / 256), 1, 1) per timestep
// The Rust driver orchestrates T dispatches with buffer swaps.

enable f64;

struct Params {
    n_states: u32,
    timestep: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> log_alpha_prev: array<f64>;
@group(0) @binding(2) var<storage, read_write> log_alpha_next: array<f64>;
@group(0) @binding(3) var<storage, read> log_trans: array<f64>;
@group(0) @binding(4) var<storage, read> log_emit: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let s = params.n_states;
    if (j >= s) { return; }

    // logsumexp over all source states i
    var max_val = f64(-1e300);
    for (var i = 0u; i < s; i++) {
        let v = log_alpha_prev[i] + log_trans[i * s + j];
        if (v > max_val) { max_val = v; }
    }

    var sum_exp = f64(0.0);
    for (var i = 0u; i < s; i++) {
        let v = log_alpha_prev[i] + log_trans[i * s + j];
        sum_exp += exp(v - max_val);
    }

    let lse = max_val + log(sum_exp);
    let emit_idx = params.timestep * s + j;
    log_alpha_next[j] = lse + log_emit[emit_idx];
}
