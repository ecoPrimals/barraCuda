// Pad - Add padding to tensor (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    channels: u32,
    batch_size: u32,
    pad_left: u32,
    pad_top: u32,
    pad_value: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_y = global_id.y;
    let out_x = global_id.x;

    if (out_y >= params.output_height || out_x >= params.output_width) {
        return;
    }

    let batch_idx = global_id.z / params.channels;
    let channel_idx = global_id.z % params.channels;

    if (batch_idx >= params.batch_size) {
        return;
    }

    // Calculate input coordinates
    let in_y = i32(out_y) - i32(params.pad_top);
    let in_x = i32(out_x) - i32(params.pad_left);

    let output_idx = ((batch_idx * params.channels + channel_idx) * params.output_height + out_y) * params.output_width + out_x;

    // Check if we're in the padded region
    if (in_y < 0 || in_y >= i32(params.input_height) || in_x < 0 || in_x >= i32(params.input_width)) {
        output[output_idx] = params.pad_value;
    } else {
        let input_idx = ((batch_idx * params.channels + channel_idx) * params.input_height + u32(in_y)) * params.input_width + u32(in_x);
        output[output_idx] = input[input_idx];
    }
}
