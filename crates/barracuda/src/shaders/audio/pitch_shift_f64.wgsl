// PitchShift - Pitch shifting without tempo change
// Changes pitch by resampling in frequency domain
// Input: Signal [length]
// Output: Shifted signal [output_length]

struct Params {
    input_length: u32,
    output_length: u32,
    rate: f32,  // Resampling rate
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.output_length) {
        return;
    }
    
    let src_pos = f64(idx) * f64(params.rate);
    let src_idx = u32(src_pos);
    let frac = src_pos - f64(src_idx);
    
    if (src_idx < params.input_length - 1u) {
        // Linear interpolation
        output[idx] = input[src_idx] * (1.0 - frac) + input[src_idx + 1u] * frac;
    } else if (src_idx < params.input_length) {
        output[idx] = input[src_idx];
    } else {
        output[idx] = 0.0;
    }
}
