// SPDX-License-Identifier: AGPL-3.0-or-later
//
// esn_reservoir_update_f64.wgsl — ESN reservoir update (f64)
//
// x(t+1) = (1-α) * x(t) + α * tanh(W_in * u(t) + W * x(t))
//
// Fused W_in*input + W_res*state → leaky tanh → new state
// Single dispatch replaces two matmul + element-wise ops.
//
// Provenance: wetSpring V17
//
// Bindings:
//   0: w_in    [R*I] f64 — input weights (row-major)
//   1: w_res   [R*R] f64 — reservoir weights (row-major)
//   2: input   [I]   f64 — current input vector
//   3: state   [R]   f64 — reservoir state (updated in-place)
//   4: params        uniform — [reservoir_size, input_size, _, _] + leak_rate

struct EsnParams {
    reservoir_size: u32,
    input_size: u32,
    _pad0: u32,
    _pad1: u32,
    leak_rate: f64,
}

@group(0) @binding(0) var<storage, read> w_in: array<f64>;
@group(0) @binding(1) var<storage, read> w_res: array<f64>;
@group(0) @binding(2) var<storage, read> input: array<f64>;
@group(0) @binding(3) var<storage, read_write> state: array<f64>;
@group(0) @binding(4) var<uniform> params: EsnParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let R = params.reservoir_size;
    let I = params.input_size;
    let alpha = params.leak_rate;

    if (i >= R) { return; }

    var pre: f64 = f64(0.0);

    let w_in_row = i * I;
    for (var j: u32 = 0u; j < I; j = j + 1u) {
        pre = pre + w_in[w_in_row + j] * input[j];
    }

    let w_res_row = i * R;
    for (var j: u32 = 0u; j < R; j = j + 1u) {
        pre = pre + w_res[w_res_row + j] * state[j];
    }

    // Leaky integration: state[i] = (1 - α) * state[i] + α * tanh(pre)
    state[i] = (f64(1.0) - alpha) * state[i] + alpha * tanh(pre);
}
