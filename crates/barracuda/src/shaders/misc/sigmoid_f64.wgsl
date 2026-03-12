// SPDX-License-Identifier: AGPL-3.0-only
// Sigmoid: Sigmoid activation function (f64 canonical)
// CUDA equivalent: cudnn::Activation(SIGMOID)
// Formula: sigmoid(x) = 1 / (1 + exp(-x))
// Use cases: Binary classification, gate activations (LSTM, GRU)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;

    if (gid >= arrayLength(&output)) {
        return;
    }

    let x = input[gid];

    // Numerically stable sigmoid
    if (x >= f64(0.0)) {
        let z = exp_f64(-x);
        output[gid] = f64(1.0) / (f64(1.0) + z);
    } else {
        let z = exp_f64(x);
        output[gid] = z / (f64(1.0) + z);
    }
}
