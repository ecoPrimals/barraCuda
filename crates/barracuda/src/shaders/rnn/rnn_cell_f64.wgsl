// SPDX-License-Identifier: AGPL-3.0-or-later
// rnn_cell.wgsl - RNN Cell (single timestep) (f64 canonical)
//
// Basic recurrent neural network cell computation
// h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
//
// Algorithm:
// 1. Compute input transformation: W_ih * x_t + b_ih
// 2. Compute hidden transformation: W_hh * h_{t-1} + b_hh
// 3. Sum both transformations
// 4. Apply tanh activation

struct Params {
    batch_size: u32,
    input_size: u32,
    hidden_size: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // [batch, input_size]
@group(0) @binding(1) var<storage, read> weight_ih: array<f64>;    // [hidden_size, input_size]
@group(0) @binding(2) var<storage, read> weight_hh: array<f64>;    // [hidden_size, hidden_size]
@group(0) @binding(3) var<storage, read> bias_ih: array<f64>;      // [hidden_size]
@group(0) @binding(4) var<storage, read> bias_hh: array<f64>;      // [hidden_size]
@group(0) @binding(5) var<storage, read> h_prev: array<f64>;        // [batch, hidden_size]
@group(0) @binding(6) var<storage, read_write> h_next: array<f64>; // [batch, hidden_size]
@group(0) @binding(7) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.x;

    if (b >= params.batch_size) {
        return;
    }

    // Compute new hidden state for each hidden unit
    for (var h: u32 = 0u; h < params.hidden_size; h = h + 1u) {
        var sum: f64 = 0.0;

        // Compute W_ih * x_t + b_ih
        // Weight matrix is [hidden_size, input_size], row-major
        for (var i: u32 = 0u; i < params.input_size; i = i + 1u) {
            let w_idx = h * params.input_size + i;
            let x_idx = b * params.input_size + i;
            sum = sum + weight_ih[w_idx] * input[x_idx];
        }
        sum = sum + bias_ih[h];

        // Compute W_hh * h_{t-1} + b_hh
        // Weight matrix is [hidden_size, hidden_size], row-major
        for (var hh: u32 = 0u; hh < params.hidden_size; hh = hh + 1u) {
            let w_idx = h * params.hidden_size + hh;
            let h_idx = b * params.hidden_size + hh;
            sum = sum + weight_hh[w_idx] * h_prev[h_idx];
        }
        sum = sum + bias_hh[h];

        // Apply tanh activation
        let idx = b * params.hidden_size + h;
        h_next[idx] = tanh_f64(sum);
    }
}
