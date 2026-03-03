// Avg Pool 1D - Temporal average pooling
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    input_size: u32,
    output_size: u32,
    channels: u32,
    batch_size: u32,
    kernel_size: u32,
    stride: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_elements = params.batch_size * params.channels * params.output_size;
    
    if (idx >= total_elements) {
        return;
    }
    
    // Decode indices: batch, channel, output_position
    let output_pos = idx % params.output_size;
    let channel = (idx / params.output_size) % params.channels;
    let batch = idx / (params.output_size * params.channels);
    
    // Calculate input window start
    let input_start = output_pos * params.stride;
    
    // Calculate average in window
    var sum = 0.0;
    var count = 0u;
    for (var k = 0u; k < params.kernel_size; k++) {
        let input_pos = input_start + k;
        if (input_pos < params.input_size) {
            let input_idx = batch * params.channels * params.input_size + 
                          channel * params.input_size + input_pos;
            sum += input[input_idx];
            count++;
        }
    }
    
    output[idx] = sum / f64(count);
}
