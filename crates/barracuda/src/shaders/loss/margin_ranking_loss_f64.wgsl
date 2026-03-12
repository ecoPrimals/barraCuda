// SPDX-License-Identifier: AGPL-3.0-only
// margin_ranking_loss.wgsl - Margin Ranking Loss
//
// Measures ranking loss between pairs
// Loss = max(0, -y * (x1 - x2) + margin)
//
// where y = 1 means x1 should rank higher, y = -1 means x2 should rank higher

struct Params {
    size: u32,
    margin: f64,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input1: array<f64>;       // First input
@group(0) @binding(1) var<storage, read> input2: array<f64>;       // Second input
@group(0) @binding(2) var<storage, read> target_data: array<f64>;       // 1 or -1
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // Per-pair loss
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let x1 = input1[idx];
    let x2 = input2[idx];
    let y = target_data[idx];
    
    // Loss = max(0, -y * (x1 - x2) + margin)
    let diff = x1 - x2;
    let loss = max(0.0, -y * diff + params.margin);
    
    output[idx] = loss;
}
