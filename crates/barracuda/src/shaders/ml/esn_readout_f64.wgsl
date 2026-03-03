// SPDX-License-Identifier: AGPL-3.0-or-later
//
// esn_readout_f64.wgsl — ESN readout matvec (f64)
//
// output[i] = W_out[i,:] · state  (matrix-vector product)
//
// Provenance: hotSpring v0.6.0 (Stanton-Murillo transport)
// Evolved from f32 → f64 for universal math library portability.
//
// Bindings:
//   0: w_out   [O*R] f64 — readout weights (row-major)
//   1: state   [R]   f64 — reservoir state
//   2: output  [O]   f64 — prediction
//   3: params  uniform — [reservoir_size, output_size]

struct ReadoutParams {
    reservoir_size: u32,
    output_size: u32,
}

@group(0) @binding(0) var<storage, read> w_out: array<f64>;
@group(0) @binding(1) var<storage, read> state: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: ReadoutParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let R = params.reservoir_size;
    let O = params.output_size;

    if (i >= O) { return; }

    var sum: f64 = f64(0.0);
    let row = i * R;
    for (var j: u32 = 0u; j < R; j = j + 1u) {
        sum = sum + w_out[row + j] * state[j];
    }
    output[i] = sum;
}
