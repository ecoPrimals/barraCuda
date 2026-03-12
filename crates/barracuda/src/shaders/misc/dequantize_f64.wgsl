// SPDX-License-Identifier: AGPL-3.0-only
// dequantize_f64.wgsl - Convert quantized integers to floating point (f64 canonical)
//
// Dequantization: Convert low-precision integers back to FP
// Used for inference with quantized models

struct Params {
    size: u32,
    scale: f64,
    zero_point: f64,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;     // Quantized values (as f64, will be cast to i32)
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // Dequantized floats
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Dequantize: (quantized_value - zero_point) * scale
    let quantized_int = i32(input[idx]);
    let quantized = f64(quantized_int);
    let dequantized = (quantized - params.zero_point) * params.scale;

    output[idx] = dequantized;
}
