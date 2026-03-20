// SPDX-License-Identifier: AGPL-3.0-or-later
// adaptive_max_pool1d.wgsl - Adaptive Max Pooling 1D
//
// Applies max pooling with adaptive kernel size to produce fixed output size
// Used in models like ResNet, VGG for variable input sizes

struct Params {
    batch_size: u32,
    channels: u32,
    in_length: u32,
    out_length: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;         // [B, C, L_in]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // [B, C, L_out]
@group(0) @binding(2) var<uniform> params: Params;

fn start_index(out_idx: u32, out_size: u32, in_size: u32) -> u32 {
    return (out_idx * in_size) / out_size;
}

fn end_index(out_idx: u32, out_size: u32, in_size: u32) -> u32 {
    return ((out_idx + 1u) * in_size + out_size - 1u) / out_size;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.channels * params.out_length;
    
    if (idx >= total) {
        return;
    }
    
    let b = idx / (params.channels * params.out_length);
    let c = (idx / params.out_length) % params.channels;
    let ol = idx % params.out_length;
    
    let start = start_index(ol, params.out_length, params.in_length);
    let end = end_index(ol, params.out_length, params.in_length);
    
    var max_val: f64 = -1e308; // -FLT_MAX
    for (var il: u32 = start; il < end; il = il + 1u) {
        let in_idx = b * params.channels * params.in_length +
                     c * params.in_length +
                     il;
        max_val = max(max_val, input[in_idx]);
    }
    
    output[idx] = max_val;
}
