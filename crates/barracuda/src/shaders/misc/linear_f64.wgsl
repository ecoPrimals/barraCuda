// SPDX-License-Identifier: AGPL-3.0-only
// Linear - Fully connected layer (complete parallel implementation) (f64 canonical)
// Computes y = xW^T + b
//
// Algorithm:
// Parallel matrix multiplication with bias addition

struct Params {
    batch_size: u32,
    in_features: u32,
    out_features: u32,
    has_bias: u32,   // 0 or 1
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;   // [batch, in_features]
@group(0) @binding(2) var<storage, read> weight: array<f64>;  // [out_features, in_features]
@group(0) @binding(3) var<storage, read> bias: array<f64>;    // [out_features]
@group(0) @binding(4) var<storage, read_write> output: array<f64>; // [batch, out_features]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.y;
    let out_f = global_id.x;
    
    if (b >= params.batch_size || out_f >= params.out_features) {
        return;
    }

    var sum = 0.0;
    
    // Matrix multiplication: x * W^T
    for (var in_f = 0u; in_f < params.in_features; in_f++) {
        let x_val = input[b * params.in_features + in_f];
        let w_val = weight[out_f * params.in_features + in_f];
        sum += x_val * w_val;
    }
    
    // Add bias if present
    if (params.has_bias != 0u) {
        sum += bias[out_f];
    }
    
    output[b * params.out_features + out_f] = sum;
}
