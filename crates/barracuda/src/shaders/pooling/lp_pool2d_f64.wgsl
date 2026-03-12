// SPDX-License-Identifier: AGPL-3.0-only
// Lp Pool 2D - Lp-norm pooling (f64 canonical)
// Computes (Σ |x_i|^p)^(1/p) over pooling window
//
// Special cases:
// - p=1: Average pooling (sum of absolutes)
// - p=2: L2 pooling
// - p=∞: Max pooling (limit as p→∞)

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    p: f64,          // Lp norm parameter
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z / params.channels;
    let c = global_id.z % params.channels;
    let oh = global_id.y;
    let ow = global_id.x;

    if (b >= params.batch_size || c >= params.channels ||
        oh >= params.out_height || ow >= params.out_width) {
        return;
    }

    var sum: f64 = 0.0;
    var count: f64 = 0.0;

    for (var kh = 0u; kh < params.kernel_size; kh = kh + 1u) {
        for (var kw = 0u; kw < params.kernel_size; kw = kw + 1u) {
            let ih_raw = oh * params.stride + kh;
            let iw_raw = ow * params.stride + kw;

            if (ih_raw >= params.padding && ih_raw < params.in_height + params.padding &&
                iw_raw >= params.padding && iw_raw < params.in_width + params.padding) {

                let ih = ih_raw - params.padding;
                let iw = iw_raw - params.padding;

                if (ih < params.in_height && iw < params.in_width) {
                    let in_idx = ((b * params.channels + c) * params.in_height + ih) * params.in_width + iw;
                    let val = input[in_idx];

                    sum = sum + pow_f64(abs(val), params.p);
                    count = count + 1.0;
                }
            }
        }
    }

    var result: f64 = 0.0;
    if (count > 0.0) {
        result = pow_f64(sum, 1.0 / params.p);
    }

    let out_idx = ((b * params.channels + c) * params.out_height + oh) * params.out_width + ow;
    output[out_idx] = result;
}
