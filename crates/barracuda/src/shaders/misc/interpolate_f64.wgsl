// Interpolate_f64.wgsl — Resize tensor using bilinear interpolation (f64 canonical)
//
// Pure WGSL implementation (universal compute)
// Hardware-agnostic (works on any GPU/CPU via WebGPU)

struct Params {
    batch: u32,
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
    let out_x = global_id.x;
    let out_y = global_id.y;

    if (out_x >= params.out_width || out_y >= params.out_height) {
        return;
    }

    let scale_x = f64(params.in_width) / f64(params.out_width);
    let scale_y = f64(params.in_height) / f64(params.out_height);

    let in_x = (f64(out_x) + 0.5) * scale_x - 0.5;
    let in_y = (f64(out_y) + 0.5) * scale_y - 0.5;

    let x0 = u32(floor(in_x));
    let y0 = u32(floor(in_y));
    let x1 = min(x0 + 1u, params.in_width - 1u);
    let y1 = min(y0 + 1u, params.in_height - 1u);

    let wx = in_x - floor(in_x);
    let wy = in_y - floor(in_y);

    for (var b = 0u; b < params.batch; b = b + 1u) {
        for (var c = 0u; c < params.channels; c = c + 1u) {
            let base_in = b * params.channels * params.in_height * params.in_width +
                          c * params.in_height * params.in_width;

            let idx00 = base_in + y0 * params.in_width + x0;
            let idx01 = base_in + y0 * params.in_width + x1;
            let idx10 = base_in + y1 * params.in_width + x0;
            let idx11 = base_in + y1 * params.in_width + x1;

            let v00 = input[idx00];
            let v01 = input[idx01];
            let v10 = input[idx10];
            let v11 = input[idx11];

            let v0 = v00 * (1.0 - wx) + v01 * wx;
            let v1 = v10 * (1.0 - wx) + v11 * wx;
            let value = v0 * (1.0 - wy) + v1 * wy;

            let out_idx = b * params.channels * params.out_height * params.out_width +
                          c * params.out_height * params.out_width +
                          out_y * params.out_width + out_x;

            output[out_idx] = value;
        }
    }
}
