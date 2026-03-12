// SPDX-License-Identifier: AGPL-3.0-only
// Clip Grad Value - Element-wise gradient clipping (f64 canonical)
// Clamps each gradient element to [-clip_value, clip_value]
//
// Algorithm:
// g_clipped = clamp(g, -clip_value, clip_value)

struct Params {
    size: u32,
    clip_value: f64,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradients: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let grad = gradients[idx];
    output[idx] = clamp(grad, -params.clip_value, params.clip_value);
}
