// SPDX-License-Identifier: AGPL-3.0-only
// U32 to F32 Conversion - Convert u32 indices to f64 for Tensor compatibility (f64 canonical)
// Converts u32 buffer to f64 buffer (for operations that output indices)
// Downcast produces f32 for GPU

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<u32>;  // u32 input
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // f64 output (downcast to f32)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    
    // Convert u32 to f64
    output[idx] = f64(input[idx]);
}
