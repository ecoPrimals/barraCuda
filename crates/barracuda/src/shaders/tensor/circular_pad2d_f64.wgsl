// circular_pad2d.wgsl - Circular/wrap padding for 2D tensors (f64 canonical)
//
// Wraps edges around (toroidal topology)
// Useful for periodic boundary conditions

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    pad_top: u32,
    pad_left: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;     // [B, C, H_in, W_in]
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // [B, C, H_out, W_out]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;

    if (c >= params.channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }

    // Calculate input position (before padding)
    let ih_raw = i32(oh) - i32(params.pad_top);
    let iw_raw = i32(ow) - i32(params.pad_left);

    // Wrap around (modulo operation for circular padding)
    var ih = ih_raw % i32(params.in_height);
    var iw = iw_raw % i32(params.in_width);

    // Handle negative modulo
    if (ih < 0) { ih = ih + i32(params.in_height); }
    if (iw < 0) { iw = iw + i32(params.in_width); }

    let in_idx = b * params.channels * params.in_height * params.in_width +
                 c * params.in_height * params.in_width +
                 u32(ih) * params.in_width +
                 u32(iw);

    let out_idx = b * params.channels * params.out_height * params.out_width +
                  c * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;

    output[out_idx] = input[in_idx];
}
