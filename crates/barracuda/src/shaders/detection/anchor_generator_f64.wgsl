// SPDX-License-Identifier: AGPL-3.0-or-later
// Anchor Generator - Generate anchor boxes (f64 canonical)

struct Params {
    feature_h: u32,
    feature_w: u32,
    stride: u32,
    num_sizes: u32,
    num_ratios: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> anchors: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read> sizes: array<f64>;
@group(0) @binding(3) var<storage, read> aspect_ratios: array<f64>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let h = global_id.y;
    let w = global_id.x;
    
    if (h >= params.feature_h || w >= params.feature_w) {
        return;
    }
    
    let cx = f64(w * params.stride) + f64(params.stride) * 0.5;
    let cy = f64(h * params.stride) + f64(params.stride) * 0.5;
    
    var anchor_idx = (h * params.feature_w + w) * params.num_sizes * params.num_ratios;
    
    for (var s = 0u; s < params.num_sizes; s = s + 1u) {
        let size = sizes[s];
        for (var r = 0u; r < params.num_ratios; r = r + 1u) {
            let ratio = aspect_ratios[r];
            let anchor_w = size * sqrt_f64(ratio);
            let anchor_h = size / sqrt_f64(ratio);
            
            anchors[anchor_idx * 4u] = cx - anchor_w * 0.5;
            anchors[anchor_idx * 4u + 1u] = cy - anchor_h * 0.5;
            anchors[anchor_idx * 4u + 2u] = cx + anchor_w * 0.5;
            anchors[anchor_idx * 4u + 3u] = cy + anchor_h * 0.5;
            
            anchor_idx = anchor_idx + 1u;
        }
    }
}
