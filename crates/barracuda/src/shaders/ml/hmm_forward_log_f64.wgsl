// SPDX-License-Identifier: AGPL-3.0-or-later
//
// hmm_forward_log_f64.wgsl — HMM Forward Pass, Log-Domain (f64)
//
// Single forward step for Hidden Markov Models. Each thread handles one
// destination state j, computing logsumexp over all source states.
//
// Max-subtract trick: logsumexp(x) = max(x) + log(Σ exp(x - max(x)))
//
// Evolved from f32 → f64 for universal math library portability.
// Reference: Rabiner (1989), Proc IEEE 77:257

@group(0) @binding(0) var<storage, read> alpha_prev: array<f64>;
@group(0) @binding(1) var<storage, read> log_trans: array<f64>;
@group(0) @binding(2) var<storage, read> log_emit: array<f64>;
@group(0) @binding(3) var<storage, read_write> alpha_curr: array<f64>;

struct HmmParams {
    n_states: u32,
}
@group(0) @binding(4) var<uniform> params: HmmParams;

@compute @workgroup_size(256)
fn hmm_forward_log(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    let n = params.n_states;
    if j >= n {
        return;
    }

    var max_val: f64 = f64(-1.7976931348623157e+308);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let v = alpha_prev[i] + log_trans[i * n + j];
        max_val = max(max_val, v);
    }

    var sum_exp: f64 = f64(0.0);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let v = alpha_prev[i] + log_trans[i * n + j];
        sum_exp = sum_exp + exp(v - max_val);
    }

    alpha_curr[j] = max_val + log(sum_exp) + log_emit[j];
}
