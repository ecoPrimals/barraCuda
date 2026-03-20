// SPDX-License-Identifier: AGPL-3.0-or-later
// gru_cell.wgsl - GRU Cell (single timestep) (f64 canonical)
//
// Gated Recurrent Unit cell computation
// r_t = σ(W_r*x_t + U_r*h_{t-1} + b_r)  (reset gate)
// z_t = σ(W_z*x_t + U_z*h_{t-1} + b_z)  (update gate)
// n_t = tanh(W_n*x_t + r_t ⊙ (U_n*h_{t-1}) + b_n)  (new gate)
// h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

struct Params {
    batch_size: u32,
    input_size: u32,
    hidden_size: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight_ih: array<f64>;
@group(0) @binding(2) var<storage, read> weight_hh: array<f64>;
@group(0) @binding(3) var<storage, read> bias_ih: array<f64>;
@group(0) @binding(4) var<storage, read> bias_hh: array<f64>;
@group(0) @binding(5) var<storage, read> h_prev: array<f64>;
@group(0) @binding(6) var<storage, read_write> h_next: array<f64>;
@group(0) @binding(7) var<uniform> params: Params;

fn sigmoid_f64(x: f64) -> f64 {
    return 1.0 / (1.0 + exp_f64(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    for (var h: u32 = 0u; h < params.hidden_size; h = h + 1u) {
        var r: f64 = 0.0;
        for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
            let w_idx = h * params.input_size + i;
            r = r + weight_ih[w_idx] * input[b * params.input_size + i];
        }
        for (var hh: u32 = 0u; hh < params.hidden_size; hh = hh + 1u) {
            let w_idx = h * params.hidden_size + hh;
            r = r + weight_hh[w_idx] * h_prev[b * params.hidden_size + hh];
        }
        r = sigmoid_f64(r + bias_ih[h] + bias_hh[h]);

        var z: f64 = 0.0;
        for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
            let w_idx = params.hidden_size * params.input_size + h * params.input_size + i;
            z = z + weight_ih[w_idx] * input[b * params.input_size + i];
        }
        for (var hh: u32 = 0u; hh < params.hidden_size; hh = hh + 1u) {
            let w_idx = params.hidden_size * params.hidden_size + h * params.hidden_size + hh;
            z = z + weight_hh[w_idx] * h_prev[b * params.hidden_size + hh];
        }
        z = sigmoid_f64(z + bias_ih[params.hidden_size + h] + bias_hh[params.hidden_size + h]);

        var n: f64 = 0.0;
        for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
            let w_idx = 2u * params.hidden_size * params.input_size + h * params.input_size + i;
            n = n + weight_ih[w_idx] * input[b * params.input_size + i];
        }
        for (var hh: u32 = 0u; hh < params.hidden_size; hh = hh + 1u) {
            let w_idx = 2u * params.hidden_size * params.hidden_size + h * params.hidden_size + hh;
            n = n + r * weight_hh[w_idx] * h_prev[b * params.hidden_size + hh];
        }
        n = tanh_f64(n + bias_ih[2u * params.hidden_size + h] + bias_hh[2u * params.hidden_size + h]);

        let h_idx = b * params.hidden_size + h;
        h_next[h_idx] = (1.0 - z) * n + z * h_prev[h_idx];
    }
}
