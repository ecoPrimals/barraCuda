// Circular Pad - Pad tensor with circular wrapping (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    batch: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    pad_top: u32,
    pad_left: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;

    if (out_x >= params.out_width || out_y >= params.out_height) {
        return;
    }

    // Calculate input coordinates with circular wrapping
    var x_offset = i32(out_x) - i32(params.pad_left);
    var y_offset = i32(out_y) - i32(params.pad_top);

    // Wrap to [0, n) using conditional (avoids % sign variance across implementations)
    let in_w = i32(params.in_width);
    let in_h = i32(params.in_height);
    while (x_offset < 0) { x_offset += in_w; }
    while (x_offset >= in_w) { x_offset -= in_w; }
    while (y_offset < 0) { y_offset += in_h; }
    while (y_offset >= in_h) { y_offset -= in_h; }
    let in_x = x_offset;
    let in_y = y_offset;

    // Process all batches and channels
    for (var b = 0u; b < params.batch; b = b + 1u) {
        for (var c = 0u; c < params.channels; c = c + 1u) {
            let in_idx = b * params.channels * params.in_height * params.in_width +
                         c * params.in_height * params.in_width +
                         u32(in_y) * params.in_width + u32(in_x);

            let out_idx = b * params.channels * params.out_height * params.out_width +
                          c * params.out_height * params.out_width +
                          out_y * params.out_width + out_x;

            output[out_idx] = input[in_idx];
        }
    }
}
