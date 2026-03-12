// SPDX-License-Identifier: AGPL-3.0-only
// LayerScale - Per-layer learnable scaling (f64 canonical)
// Used in vision transformers (CaiT, LeViT) to stabilize training
//
// Algorithm: LayerScale(x) = gamma ⊙ x
// Element-wise multiplication with learnable per-channel parameters

struct Params {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;  // Per-channel scaling factors
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Element-wise scaling: output = input * gamma
    output[idx] = input[idx] * gamma[idx];
}
