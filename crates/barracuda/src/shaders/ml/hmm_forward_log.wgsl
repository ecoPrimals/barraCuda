// hmm_forward_log.wgsl — HMM Forward Pass, Log-Domain (f32)
//
// Computes a single forward step for Hidden Markov Models on GPU
// using log-domain arithmetic to avoid underflow. Each thread handles
// one destination state j, computing logsumexp over all source states.
//
// Uses the max-subtract trick for numerical stability:
//   logsumexp(x) = max(x) + log(Σ exp(x - max(x)))
//
// Provenance: neuralSpring metalForge (Feb 21, 2026) → ToadStool absorption
// Reference: Rabiner (1989), "A Tutorial on HMM", Proc IEEE 77:257

@group(0) @binding(0) var<storage, read> alpha_prev: array<f32>;
@group(0) @binding(1) var<storage, read> log_trans: array<f32>;   // [S × S] row-major
@group(0) @binding(2) var<storage, read> log_emit: array<f32>;    // [S] for current obs
@group(0) @binding(3) var<storage, read_write> alpha_curr: array<f32>;

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

    var max_val: f32 = -3.4028235e+38;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let v = alpha_prev[i] + log_trans[i * n + j];
        max_val = max(max_val, v);
    }

    var sum_exp: f32 = 0.0;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let v = alpha_prev[i] + log_trans[i * n + j];
        sum_exp = sum_exp + exp(v - max_val);
    }

    alpha_curr[j] = max_val + log(sum_exp) + log_emit[j];
}
