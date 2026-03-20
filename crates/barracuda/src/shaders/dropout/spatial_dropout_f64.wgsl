// SPDX-License-Identifier: AGPL-3.0-or-later
// spatial_dropout_f64.wgsl - Spatial Dropout (Channel-wise dropout) (f64 canonical)
//
// Drops entire feature maps (channels) instead of individual elements
// More effective for convolutional networks
//
// Reference: "Efficient Object Localization Using Convolutional Networks" by Tompson et al.

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    drop_prob: f64,     // Dropout probability
    training: u32,      // 1 = training mode, 0 = inference mode
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;         // [B, C, H, W]
@group(0) @binding(1) var<storage, read> mask: array<f64>;          // [B, C] - channel mask
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // [B, C, H, W]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let h = global_id.y;
    let w = global_id.x;
    
    if (c >= params.channels || h >= params.height || w >= params.width) {
        return;
    }
    
    let idx = b * params.channels * params.height * params.width +
              c * params.height * params.width +
              h * params.width +
              w;
    
    if (params.training == 0u) {
        // Inference mode: pass through
        output[idx] = input[idx];
    } else {
        // Training mode: apply channel mask
        let mask_idx = b * params.channels + c;
        let channel_mask = mask[mask_idx];
        
        // Scale by 1/(1-p) to maintain expected value
        let scale = 1.0 / (1.0 - params.drop_prob);
        output[idx] = input[idx] * channel_mask * scale;
    }
}
