// SPDX-License-Identifier: AGPL-3.0-only
// lstm_cell.wgsl - LSTM Cell (single timestep) (f64 canonical)
//
// Long Short-Term Memory cell computation
// i_t = σ(W_i*x_t + U_i*h_{t-1} + b_i)
// f_t = σ(W_f*x_t + U_f*h_{t-1} + b_f)
// g_t = tanh(W_g*x_t + U_g*h_{t-1} + b_g)
// o_t = σ(W_o*x_t + U_o*h_{t-1} + b_o)
// c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
// h_t = o_t ⊙ tanh(c_t)

struct Params {
    batch_size: u32,
    input_size: u32,
    hidden_size: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight_ih: array<f64>;
@group(0) @binding(2) var<storage, read> weight_hh: array<f64>;
@group(0) @binding(3) var<storage, read> bias: array<f64>;
@group(0) @binding(4) var<storage, read_write> h_prev: array<f64>;
@group(0) @binding(5) var<storage, read_write> c_prev: array<f64>;
@group(0) @binding(6) var<storage, read_write> h_next: array<f64>;
@group(0) @binding(7) var<storage, read_write> c_next: array<f64>;
@group(0) @binding(8) var<uniform> params: Params;

fn sigmoid_f64(x: f64) -> f64 {
    return 1.0 / (1.0 + exp_f64(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    for (var h: u32 = 0u; h < params.hidden_size; h = h + 1u) {
        var gates: array<f64, 4>;

        for (var gate: u32 = 0u; gate < 4u; gate = gate + 1u) {
            var sum: f64 = 0.0;

            for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
                let w_idx = gate * params.hidden_size * params.input_size + h * params.input_size + i;
                let x_idx = b * params.input_size + i;
                sum = sum + weight_ih[w_idx] * input[x_idx];
            }

            for (var hh: u32 = 0u; hh < params.hidden_size; hh = hh + 1u) {
                let w_idx = gate * params.hidden_size * params.hidden_size + h * params.hidden_size + hh;
                let h_idx = b * params.hidden_size + hh;
                sum = sum + weight_hh[w_idx] * h_prev[h_idx];
            }

            let bias_idx = gate * params.hidden_size + h;
            sum = sum + bias[bias_idx] + bias[4u * params.hidden_size + bias_idx];

            if (gate == 2u) {
                gates[gate] = tanh_f64(sum);
            } else {
                gates[gate] = sigmoid_f64(sum);
            }
        }

        let i_gate = gates[0];
        let f_gate = gates[1];
        let g_gate = gates[2];
        let o_gate = gates[3];

        let idx = b * params.hidden_size + h;
        let c_new = f_gate * c_prev[idx] + i_gate * g_gate;
        c_next[idx] = c_new;

        h_next[idx] = o_gate * tanh_f64(c_new);
    }
}
