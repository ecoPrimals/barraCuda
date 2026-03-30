// SPDX-License-Identifier: AGPL-3.0-or-later
//
// hmm_backward_f64.wgsl — Full-pass HMM Backward Algorithm (f64)
//
// Computes the backward variable β_t(i) for all states i at timestep t
// in a single dispatch per timestep.
//
// β_t(i) = Σ_j trans(i, j) × emit(j, O_{t+1}) × β_{t+1}(j)
//
// Log-domain:
//   log β_t(i) = logsumexp_j(log_trans(i, j) + log_emit(j, O_{t+1}) + log β_{t+1}(j))
//
// Layout:
//   log_beta_next:  [S] f64  — log β at timestep t+1
//   log_beta_curr:  [S] f64  — log β at timestep t (output)
//   log_trans:      [S×S] f64 — log transition matrix (row-major: [from × to])
//   log_emit:       [T×S] f64 — log emission probs, pre-extracted per timestep
//
// Dispatch: (ceil(S / 256), 1, 1) per timestep (backwards from T-2 to 0)

struct Params {
    n_states: u32,
    next_timestep: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> log_beta_next: array<f64>;
@group(0) @binding(2) var<storage, read_write> log_beta_curr: array<f64>;
@group(0) @binding(3) var<storage, read> log_trans: array<f64>;
@group(0) @binding(4) var<storage, read> log_emit: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let s = params.n_states;
    if (i >= s) { return; }

    // logsumexp over all destination states j
    var max_val = f64(-1e300);
    for (var j = 0u; j < s; j++) {
        let emit_idx = params.next_timestep * s + j;
        let v = log_trans[i * s + j] + log_emit[emit_idx] + log_beta_next[j];
        if (v > max_val) { max_val = v; }
    }

    var sum_exp = f64(0.0);
    for (var j = 0u; j < s; j++) {
        let emit_idx = params.next_timestep * s + j;
        let v = log_trans[i * s + j] + log_emit[emit_idx] + log_beta_next[j];
        sum_exp += exp(v - max_val);
    }

    log_beta_curr[i] = max_val + log(sum_exp);
}
