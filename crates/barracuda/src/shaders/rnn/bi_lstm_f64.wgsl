// SPDX-License-Identifier: AGPL-3.0-only
// bi_lstm.wgsl - Bidirectional LSTM (f64 canonical)
//
// Processes sequence in both forward and backward directions
// Commonly used in sequence modeling, NLP, speech recognition
//
// LSTM Cell: i_t = σ(W_i*x_t + U_i*h_{t-1} + b_i)
//            f_t = σ(W_f*x_t + U_f*h_{t-1} + b_f)
//            o_t = σ(W_o*x_t + U_o*h_{t-1} + b_o)
//            g_t = tanh(W_g*x_t + U_g*h_{t-1} + b_g)
//            c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
//            h_t = o_t ⊙ tanh(c_t)

struct Params {
    batch_size: u32,
    seq_len: u32,
    input_size: u32,
    hidden_size: u32,
    direction: u32,  // 0 = forward, 1 = backward
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight_ih: array<f64>;
@group(0) @binding(2) var<storage, read> weight_hh: array<f64>;
@group(0) @binding(3) var<storage, read> bias_ih: array<f64>;
@group(0) @binding(4) var<storage, read> bias_hh: array<f64>;
@group(0) @binding(5) var<storage, read_write> h_state: array<f64>;
@group(0) @binding(6) var<storage, read_write> c_state: array<f64>;
@group(0) @binding(7) var<storage, read_write> output: array<f64>;
@group(0) @binding(8) var<uniform> params: Params;

fn sigmoid_f64(x: f64) -> f64 {
    return 1.0 / (1.0 + exp_f64(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    let start = select(0u, params.seq_len - 1u, params.direction == 1u);
    let end = select(params.seq_len, 0u, params.direction == 1u);
    let step = select(1i, -1i, params.direction == 1u);

    var t = i32(start);
    loop {
        if (params.direction == 0u && t >= i32(end)) { break; }
        if (params.direction == 1u && t < i32(end)) { break; }

        let t_u = u32(t);

        let x_offset = t_u * params.batch_size * params.input_size + b * params.input_size;

        var gates: array<f64, 4>;
        for (var gate: u32 = 0u; gate < 4u; gate = gate + 1u) {
            var sum: f64 = 0.0;

            for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
                let w_idx = gate * params.hidden_size * params.input_size + i;
                sum = sum + weight_ih[w_idx] * input[x_offset + i];
            }

            for (var h: u32 = 0u; h < params.hidden_size; h = h + 1u) {
                let w_idx = gate * params.hidden_size * params.hidden_size + h;
                let h_idx = b * params.hidden_size + h;
                sum = sum + weight_hh[w_idx] * h_state[h_idx];
            }

            sum = sum + bias_ih[gate * params.hidden_size] + bias_hh[gate * params.hidden_size];

            gates[gate] = select(tanh_f64(sum), sigmoid_f64(sum), gate < 3u);
        }

        let i_gate = gates[0];
        let f_gate = gates[1];
        let g_gate = gates[3];
        let o_gate = gates[2];

        for (var h: u32 = 0u; h < params.hidden_size; h = h + 1u) {
            let idx = b * params.hidden_size + h;

            let c_prev = c_state[idx];
            let c_new = f_gate * c_prev + i_gate * g_gate;
            c_state[idx] = c_new;

            let h_new = o_gate * tanh_f64(c_new);
            h_state[idx] = h_new;

            let out_idx = t_u * params.batch_size * params.hidden_size + b * params.hidden_size + h;
            output[out_idx] = h_new;
        }

        t = t + step;
    }
}
