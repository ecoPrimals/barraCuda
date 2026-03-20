// SPDX-License-Identifier: AGPL-3.0-or-later
// fake_quantize_f64.wgsl - Simulate quantization for training (f64 canonical)
//
// Fake quantization simulates the effect of quantization during training
// by quantizing values to N bits but keeping them in floating point format.

struct Params {
    size: u32,
    num_bits: u32,       // Number of bits (e.g., 8 for INT8)
    scale: f64,          // Quantization scale
    zero_point: f64,     // Quantization zero point
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let value = input[idx];

    // Calculate quantization range
    let qmin = 0.0;
    let qmax = f64((1u << params.num_bits) - 1u);

    // Quantize: float -> int (simulated)
    let quantized = round((value / params.scale) + params.zero_point);

    // Clamp to valid range
    let clamped = clamp(quantized, qmin, qmax);

    // Dequantize: int -> float (keeping in FP)
    let dequantized = (clamped - params.zero_point) * params.scale;

    output[idx] = dequantized;
}
