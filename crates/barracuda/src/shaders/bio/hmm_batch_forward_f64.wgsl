// SPDX-License-Identifier: AGPL-3.0-or-later
//
// hmm_batch_forward_f64.wgsl — Batch HMM forward algorithm (f64)
//
// One thread per observation sequence. Sequential over T timesteps within
// each sequence, parallel across B sequences. All computation in log-domain
// with logsumexp for numerical stability.
//
// Binding layout matches HmmBatchForwardF64::dispatch():
//   0: uniform params
//   1: storage_read  log_trans       [S×S]
//   2: storage_read  log_emit        [S×M]
//   3: storage_read  log_pi          [S]
//   4: storage_read  observations    [B×T]  (u32 cast to f64)
//   5: storage_rw    log_alpha_out   [B×T×S]
//   6: storage_rw    log_lik_out     [B]
//
// Dispatch: (ceil(B / 256), 1, 1)

enable f64;

struct Params {
    n_states:  u32,
    n_symbols: u32,
    n_steps:   u32,
    n_seqs:    u32,
}

@group(0) @binding(0) var<uniform>           params:       Params;
@group(0) @binding(1) var<storage, read>     log_trans:    array<f64>;
@group(0) @binding(2) var<storage, read>     log_emit:     array<f64>;
@group(0) @binding(3) var<storage, read>     log_pi:       array<f64>;
@group(0) @binding(4) var<storage, read>     observations: array<f64>;
@group(0) @binding(5) var<storage, read_write> log_alpha_out: array<f64>;
@group(0) @binding(6) var<storage, read_write> log_lik_out:   array<f64>;

fn logsumexp_slice(base: u32, len: u32) -> f64 {
    var mx = f64(-1e300);
    for (var i = 0u; i < len; i = i + 1u) {
        let v = log_alpha_out[base + i];
        if v > mx { mx = v; }
    }
    if mx <= f64(-1e299) { return f64(-1e300); }
    var s = f64(0.0);
    for (var i = 0u; i < len; i = i + 1u) {
        s = s + exp(log_alpha_out[base + i] - mx);
    }
    return mx + log(s);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq = gid.x;
    if seq >= params.n_seqs { return; }

    let s = params.n_states;
    let t_total = params.n_steps;
    let alpha_base = seq * t_total * s;

    // t = 0: log_alpha(seq, 0, j) = log_pi(j) + log_emit(j, obs[seq, 0])
    let obs0 = u32(observations[seq * t_total]);
    for (var j = 0u; j < s; j = j + 1u) {
        let emit_val = log_emit[j * params.n_symbols + obs0];
        log_alpha_out[alpha_base + j] = log_pi[j] + emit_val;
    }

    // t = 1..T-1
    for (var t = 1u; t < t_total; t = t + 1u) {
        let obs_t = u32(observations[seq * t_total + t]);
        let prev_off = alpha_base + (t - 1u) * s;
        let curr_off = alpha_base + t * s;

        for (var j = 0u; j < s; j = j + 1u) {
            // logsumexp over source states
            var mx = f64(-1e300);
            for (var i = 0u; i < s; i = i + 1u) {
                let v = log_alpha_out[prev_off + i] + log_trans[i * s + j];
                if v > mx { mx = v; }
            }
            var sum_e = f64(0.0);
            for (var i = 0u; i < s; i = i + 1u) {
                let v = log_alpha_out[prev_off + i] + log_trans[i * s + j];
                sum_e = sum_e + exp(v - mx);
            }
            let lse = mx + log(sum_e);
            log_alpha_out[curr_off + j] = lse + log_emit[j * params.n_symbols + obs_t];
        }
    }

    // log-likelihood: logsumexp of final alpha
    let final_off = alpha_base + (t_total - 1u) * s;
    log_lik_out[seq] = logsumexp_slice(final_off, s);
}
