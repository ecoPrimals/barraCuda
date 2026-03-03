// hmm_forward_f64.wgsl — HMM Batch Forward Algorithm (f64)
//
// Runs the forward algorithm for B independent observation sequences
// through the same HMM model with S states and K emission symbols.
//
// One thread per sequence. Sequential over T time steps within each
// sequence, parallel across sequences.
//
// GPU dispatch: ceil(B / 256) workgroups, 256 threads each.
//
// Provenance: wetSpring handoff v5 (Feb 20, 2026) → ToadStool absorption

struct HmmParams {
    n_states:  u32,
    n_symbols: u32,
    n_steps:   u32,
    n_seqs:    u32,
}

@group(0) @binding(0) var<uniform>             params:        HmmParams;
@group(0) @binding(1) var<storage, read>       log_trans:     array<f64>;  // [S × S] row-major
@group(0) @binding(2) var<storage, read>       log_emit:      array<f64>;  // [S × K] row-major
@group(0) @binding(3) var<storage, read>       log_pi:        array<f64>;  // [S]
@group(0) @binding(4) var<storage, read>       observations:  array<u32>;  // [B × T]
@group(0) @binding(5) var<storage, read_write> log_alpha_out: array<f64>;  // [B × T × S]
@group(0) @binding(6) var<storage, read_write> log_lik_out:   array<f64>;  // [B]

fn log_sum_exp2(a: f64, b: f64) -> f64 {
    if a < -1.0e300 { return b; }
    if b < -1.0e300 { return a; }
    let mx = max(a, b);
    return mx + log(exp(a - mx) + exp(b - mx));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let seq_idx = gid.x;
    if seq_idx >= params.n_seqs { return; }

    let S = params.n_states;
    let T = params.n_steps;
    let alpha_base = seq_idx * T * S;
    let obs_base = seq_idx * T;

    // Initialization: alpha[0][i] = log_pi[i] + log_emit[i][obs[0]]
    let obs0 = observations[obs_base];
    for (var i: u32 = 0u; i < S; i = i + 1u) {
        let lp = log_pi[i];
        let le = log_emit[i * params.n_symbols + obs0];
        log_alpha_out[alpha_base + i] = lp + le;
    }

    // Induction: alpha[t][j] = log_emit[j][obs[t]] + logsumexp_i(alpha[t-1][i] + log_trans[i][j])
    for (var t: u32 = 1u; t < T; t = t + 1u) {
        let obs_t = observations[obs_base + t];
        for (var j: u32 = 0u; j < S; j = j + 1u) {
            var acc: f64 = -1.0e300;
            for (var i: u32 = 0u; i < S; i = i + 1u) {
                let prev = log_alpha_out[alpha_base + (t - 1u) * S + i];
                let lt = log_trans[i * S + j];
                acc = log_sum_exp2(acc, prev + lt);
            }
            let le = log_emit[j * params.n_symbols + obs_t];
            log_alpha_out[alpha_base + t * S + j] = acc + le;
        }
    }

    // Termination: log_lik = logsumexp_i(alpha[T-1][i])
    var log_lik: f64 = -1.0e300;
    for (var i: u32 = 0u; i < S; i = i + 1u) {
        let val = log_alpha_out[alpha_base + (T - 1u) * S + i];
        log_lik = log_sum_exp2(log_lik, val);
    }
    log_lik_out[seq_idx] = log_lik;
}
