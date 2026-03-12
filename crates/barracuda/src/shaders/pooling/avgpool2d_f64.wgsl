// SPDX-License-Identifier: AGPL-3.0-only
// AvgPool2D - Average pooling (2D) with padding support
//
// Supports configurable pool_size, stride, and padding (pad_h, pad_w).
// Out-of-bounds (padded) positions contribute 0 and are excluded from count.

struct AvgPool2DParams {
    input_width: u32,
    input_height: u32,
    pool_size: u32,
    stride: u32,
    pad_h: u32,
    pad_w: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: AvgPool2DParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;

    let output_width = (params.input_width + 2u * params.pad_w - params.pool_size) / params.stride + 1u;
    let output_height = (params.input_height + 2u * params.pad_h - params.pool_size) / params.stride + 1u;

    if (out_x >= output_width || out_y >= output_height) {
        return;
    }

    let in_y_start = i32(out_y * params.stride) - i32(params.pad_h);
    let in_x_start = i32(out_x * params.stride) - i32(params.pad_w);

    var sum = 0.0;
    var count = 0u;

    for (var dy = 0u; dy < params.pool_size; dy = dy + 1u) {
        for (var dx = 0u; dx < params.pool_size; dx = dx + 1u) {
            let y = in_y_start + i32(dy);
            let x = in_x_start + i32(dx);

            if (y >= 0 && y < i32(params.input_height) && x >= 0 && x < i32(params.input_width)) {
                let idx = u32(y) * params.input_width + u32(x);
                sum = sum + input[idx];
                count = count + 1u;
            }
        }
    }

    let out_idx = out_y * output_width + out_x;
    output[out_idx] = sum / f64(count);
}
