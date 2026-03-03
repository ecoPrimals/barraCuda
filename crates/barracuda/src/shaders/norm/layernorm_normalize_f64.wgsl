// LayerNorm Normalization (Dispatch 2 of 2) - f64 canonical
//
// **2-DISPATCH FUSED LAYERNORM - PART 2: NORMALIZATION**
//
// This shader reads global statistics computed in dispatch 1 and normalizes
// all elements in a single pass with gamma and beta application.
//
// Algorithm:
//   1. Read global mean and variance from buffer (computed in dispatch 1)
//   2. Normalize each element: (x - mean) / sqrt(variance + epsilon)
//   3. Apply gamma (scale) and beta (shift): output = normalized * gamma + beta
//
// Combined with dispatch 1, this eliminates 1/3 launch overhead vs original
// 3-pass implementation (3 dispatches → 2 dispatches).

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;
@group(0) @binding(2) var<storage, read> beta: array<f64>;
@group(0) @binding(3) var<storage, read> global_stats: array<f64>;  // [mean, variance]
@group(0) @binding(4) var<storage, read_write> output: array<f64>;

struct Params {
    size: u32,
    epsilon: f64,
}
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Read global statistics (computed in dispatch 1)
    let mean = global_stats[0];
    let variance = global_stats[1];
    let std_dev = sqrt_f64(variance + params.epsilon);

    // Normalize
    let value = input[idx];
    let normalized = (value - mean) / std_dev;

    // Apply gamma and beta
    output[idx] = normalized * gamma[idx] + beta[idx];
}
