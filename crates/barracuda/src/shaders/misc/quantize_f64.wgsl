// quantize_f64.wgsl - Convert FP32 to INT8/INT4 quantization (f64 canonical)
//
// Quantization: Convert floating point values to low-precision integers
// Used for model compression and efficient inference
//
// Algorithm: q = round((x - zero_point) * scale)
// For INT8: clamp to [-128, 127]
// For INT4: clamp to [-8, 7]

struct Params {
    size: u32,
    scale: f64,
    zero_point: f64,
    num_bits: u32,      // 4 for INT4, 8 for INT8
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;  // Quantized integers
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    let value = input[idx];
    
    // Quantize: round((x - zero_point) * scale)
    // Note: scale here is 1/quantization_scale (inverse scale)
    let quantized = round((value - params.zero_point) * params.scale);
    
    // Calculate quantization range based on bits
    var qmin: f64;
    var qmax: f64;
    if (params.num_bits == 4u) {
        qmin = -8.0;
        qmax = 7.0;
    } else {
        // INT8 (default)
        qmin = -128.0;
        qmax = 127.0;
    }
    
    // Clamp to valid range
    let clamped = clamp(quantized, qmin, qmax);
    
    // Convert to integer
    output[idx] = i32(clamped);
}
