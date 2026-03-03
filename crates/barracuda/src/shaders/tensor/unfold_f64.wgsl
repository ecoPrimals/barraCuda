// Unfold - Extract sliding windows (im2col) (f64 canonical)
// Extracts sliding local blocks from input tensor
// Used for efficient convolution implementation
//
// Example: unfold([B, C, H, W], kernel_size=3, stride=1) → [B, C*K*K, L]
//   where L = number of blocks

struct Params {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    out_height: u32,  // Number of blocks in height
    out_width: u32,   // Number of blocks in width
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;  // [B, C, H, W]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [B, C*K_h*K_w, L]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z;
    let block_h = global_id.y;
    let block_w = global_id.x;

    if (b >= params.batch_size || block_h >= params.out_height || block_w >= params.out_width) {
        return;
    }

    let block_idx = block_h * params.out_width + block_w;

    // Extract sliding window
    for (var c = 0u; c < params.channels; c++) {
        for (var kh = 0u; kh < params.kernel_height; kh++) {
            for (var kw = 0u; kw < params.kernel_width; kw++) {
                // Compute input position with dilation and padding
                let ih_base = block_h * params.stride;
                let iw_base = block_w * params.stride;
                let ih = ih_base + kh * params.dilation;
                let iw = iw_base + kw * params.dilation;

                var value = 0.0;

                // Check if within padded bounds
                if (ih >= params.padding && ih < params.in_height + params.padding &&
                    iw >= params.padding && iw < params.in_width + params.padding) {

                    let ih_actual = ih - params.padding;
                    let iw_actual = iw - params.padding;

                    if (ih_actual < params.in_height && iw_actual < params.in_width) {
                        let in_idx = ((b * params.channels + c) * params.in_height + ih_actual) * params.in_width + iw_actual;
                        value = input[in_idx];
                    }
                }

                // Output layout: [B, C*K_h*K_w, L]
                let channel_offset = (c * params.kernel_height + kh) * params.kernel_width + kw;
                let out_idx = (b * (params.channels * params.kernel_height * params.kernel_width) + channel_offset)
                              * (params.out_height * params.out_width) + block_idx;
                output[out_idx] = value;
            }
        }
    }
}
