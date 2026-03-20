// SPDX-License-Identifier: AGPL-3.0-or-later
// fractional_max_pool2d.wgsl - Fractional max pooling
//
// Stochastic pooling with non-integer pooling ratios
// Improves generalization by introducing randomness
//
// Reference: "Fractional Max-Pooling" by Graham (2014)

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;     // [B, C, H_in, W_in]
@group(0) @binding(1) var<storage, read> pool_seq_h: array<u32>; // Pooling sequence for height
@group(0) @binding(2) var<storage, read> pool_seq_w: array<u32>; // Pooling sequence for width
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [B, C, H_out, W_out]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (c >= params.channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    // Get pooling region from sequences
    let h_start = pool_seq_h[oh];
    let h_end = pool_seq_h[oh + 1u];
    let w_start = pool_seq_w[ow];
    let w_end = pool_seq_w[ow + 1u];
    
    var max_val: f64 = -1e308;
    
    // Find max in pooling region
    for (var h: u32 = h_start; h < h_end; h = h + 1u) {
        for (var w: u32 = w_start; w < w_end; w = w + 1u) {
            let in_idx = b * params.channels * params.in_height * params.in_width +
                         c * params.in_height * params.in_width +
                         h * params.in_width +
                         w;
            
            max_val = max(max_val, input[in_idx]);
        }
    }
    
    let out_idx = b * params.channels * params.out_height * params.out_width +
                  c * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = max_val;
}
