// SPDX-License-Identifier: AGPL-3.0-only
// Grid Sample_f64.wgsl — Spatial transformer network sampling (f64 canonical)
// Samples input at arbitrary grid positions using bilinear interpolation

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> grid: array<f64>;  // [B, H_out, W_out, 2] normalized coords
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_y = global_id.y;
    let out_x = global_id.x;

    if (out_y >= params.out_height || out_x >= params.out_width) {
        return;
    }

    let h = params.in_height;
    let w = params.in_width;

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        let grid_idx = ((b * params.out_height + out_y) * params.out_width + out_x) * 2u;
        let grid_x = grid[grid_idx];
        let grid_y = grid[grid_idx + 1u];

        for (var c = 0u; c < params.channels; c = c + 1u) {
            let x_pix = (grid_x + 1.0) * f64(w - 1u) * 0.5;
            let y_pix = (grid_y + 1.0) * f64(h - 1u) * 0.5;

            let x0 = u32(floor(x_pix));
            let y0 = u32(floor(y_pix));
            let x1 = min(x0 + 1u, w - 1u);
            let y1 = min(y0 + 1u, h - 1u);

            let wx = x_pix - f64(x0);
            let wy = y_pix - f64(y0);

            var sampled_value: f64 = 0.0;
            if (x0 < w && y0 < h) {
                let base_idx = (b * params.channels + c) * h * w;
                let v00 = input[base_idx + y0 * w + x0];
                let v01 = input[base_idx + y0 * w + x1];
                let v10 = input[base_idx + y1 * w + x0];
                let v11 = input[base_idx + y1 * w + x1];
                let v0 = v00 * (1.0 - wx) + v01 * wx;
                let v1 = v10 * (1.0 - wx) + v11 * wx;
                sampled_value = v0 * (1.0 - wy) + v1 * wy;
            }

            let out_idx = ((b * params.channels + c) * params.out_height + out_y)
                          * params.out_width + out_x;
            output[out_idx] = sampled_value;
        }
    }
}
