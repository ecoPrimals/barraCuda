// Scaled Dot-Product Attention in DF64 precision
// attn(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
//
// Single-head, batch_size=1, for validation / high-precision inference.
// Q: [seq_q, d_k], K: [seq_k, d_k], V: [seq_k, d_v]
// Output: [seq_q, d_v]
//
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

struct SdpaParams {
    seq_q: u32,
    seq_k: u32,
    d_k: u32,
    d_v: u32,
}

@group(0) @binding(0) var<storage, read> q_hi: array<f32>;
@group(0) @binding(1) var<storage, read> q_lo: array<f32>;
@group(0) @binding(2) var<storage, read> k_hi: array<f32>;
@group(0) @binding(3) var<storage, read> k_lo: array<f32>;
@group(0) @binding(4) var<storage, read> v_hi: array<f32>;
@group(0) @binding(5) var<storage, read> v_lo: array<f32>;
@group(0) @binding(6) var<storage, read_write> out_hi: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_lo: array<f32>;
@group(0) @binding(8) var<uniform> params: SdpaParams;

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>) {
    let qi = wg_id.x;
    if (qi >= params.seq_q) { return; }

    let inv_sqrt_dk = df64_div(df64_from_f32(1.0), sqrt_df64(df64_from_f32(f32(params.d_k))));

    // Compute attention scores: score[j] = (Q[qi] . K[j]) / sqrt(d_k)
    // Then softmax over j, then weighted sum of V[j]

    // Pass 1: compute scores and find max
    var max_score = Df64(-3.4e38, 0.0);
    for (var j = 0u; j < params.seq_k; j = j + 1u) {
        var dot = df64_from_f32(0.0);
        for (var d = 0u; d < params.d_k; d = d + 1u) {
            let q_val = Df64(q_hi[qi * params.d_k + d], q_lo[qi * params.d_k + d]);
            let k_val = Df64(k_hi[j * params.d_k + d], k_lo[j * params.d_k + d]);
            dot = df64_add(dot, df64_mul(q_val, k_val));
        }
        let score = df64_mul(dot, inv_sqrt_dk);
        if (score.hi > max_score.hi) {
            max_score = score;
        }
    }

    // Pass 2: exp(score - max) and sum
    var sum_exp = df64_from_f32(0.0);
    for (var j = 0u; j < params.seq_k; j = j + 1u) {
        var dot = df64_from_f32(0.0);
        for (var d = 0u; d < params.d_k; d = d + 1u) {
            let q_val = Df64(q_hi[qi * params.d_k + d], q_lo[qi * params.d_k + d]);
            let k_val = Df64(k_hi[j * params.d_k + d], k_lo[j * params.d_k + d]);
            dot = df64_add(dot, df64_mul(q_val, k_val));
        }
        let score = df64_mul(dot, inv_sqrt_dk);
        let exp_val = exp_df64(df64_sub(score, max_score));
        sum_exp = df64_add(sum_exp, exp_val);
    }

    // Pass 3: output = sum_j( softmax(score_j) * V[j] )
    for (var dv = 0u; dv < params.d_v; dv = dv + 1u) {
        var acc = df64_from_f32(0.0);
        for (var j = 0u; j < params.seq_k; j = j + 1u) {
            var dot = df64_from_f32(0.0);
            for (var d = 0u; d < params.d_k; d = d + 1u) {
                let q_val = Df64(q_hi[qi * params.d_k + d], q_lo[qi * params.d_k + d]);
                let k_val = Df64(k_hi[j * params.d_k + d], k_lo[j * params.d_k + d]);
                dot = df64_add(dot, df64_mul(q_val, k_val));
            }
            let score = df64_mul(dot, inv_sqrt_dk);
            let weight = df64_div(exp_df64(df64_sub(score, max_score)), sum_exp);
            let v_val = Df64(v_hi[j * params.d_v + dv], v_lo[j * params.d_v + dv]);
            acc = df64_add(acc, df64_mul(weight, v_val));
        }
        out_hi[qi * params.d_v + dv] = acc.hi;
        out_lo[qi * params.d_v + dv] = acc.lo;
    }
}
