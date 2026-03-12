// SPDX-License-Identifier: AGPL-3.0-only
// Reflection Pad - Pad tensor with reflection at boundaries (f64 canonical)
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

    // Calculate input coordinates with reflection
    var in_x: i32;
    var in_y: i32;

    // Reflect horizontally
    let x_offset = i32(out_x) - i32(params.pad_left);
    if (x_offset < 0) {
        in_x = -x_offset;
    } else if (x_offset >= i32(params.in_width)) {
        in_x = i32(params.in_width) - 2 - (x_offset - i32(params.in_width));
    } else {
        in_x = x_offset;
    }

    // Reflect vertically
    let y_offset = i32(out_y) - i32(params.pad_top);
    if (y_offset < 0) {
        in_y = -y_offset;
    } else if (y_offset >= i32(params.in_height)) {
        in_y = i32(params.in_height) - 2 - (y_offset - i32(params.in_height));
    } else {
        in_y = y_offset;
    }

    // Clamp to valid range
    in_x = clamp(in_x, 0, i32(params.in_width) - 1);
    in_y = clamp(in_y, 0, i32(params.in_height) - 1);

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
