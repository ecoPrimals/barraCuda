// Replication Pad - Pad tensor by replicating edge values (f64 canonical)
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

    // Calculate input coordinates with clamping (replication)
    let x_offset = i32(out_x) - i32(params.pad_left);
    let y_offset = i32(out_y) - i32(params.pad_top);

    let in_x = clamp(x_offset, 0, i32(params.in_width) - 1);
    let in_y = clamp(y_offset, 0, i32(params.in_height) - 1);

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
