// Interpolate Nearest_f64.wgsl — Nearest neighbor interpolation (f64 canonical)
// Resizes tensors using nearest neighbor sampling

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_y = global_id.y;
    let out_x = global_id.x;

    if (out_y >= params.out_height || out_x >= params.out_width) {
        return;
    }

    let scale_h = f64(params.in_height) / f64(params.out_height);
    let scale_w = f64(params.in_width) / f64(params.out_width);

    let in_y = u32(f64(out_y) * scale_h);
    let in_x = u32(f64(out_x) * scale_w);

    let in_y_clamped = min(in_y, params.in_height - 1u);
    let in_x_clamped = min(in_x, params.in_width - 1u);

    for (var b = 0u; b < params.batch_size; b = b + 1u) {
        for (var c = 0u; c < params.channels; c = c + 1u) {
            let in_idx = ((b * params.channels + c) * params.in_height + in_y_clamped)
                         * params.in_width + in_x_clamped;
            let out_idx = ((b * params.channels + c) * params.out_height + out_y)
                          * params.out_width + out_x;

            output[out_idx] = input[in_idx];
        }
    }
}
