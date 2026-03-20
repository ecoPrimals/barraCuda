// SPDX-License-Identifier: AGPL-3.0-or-later
// Fold - Inverse of unfold (col2im) (f64 canonical)
// Combines sliding local blocks back into tensor
// Used in transposed convolutions
//
// Example: fold([B, C*K*K, L], output_size=[H, W]) → [B, C, H, W]

struct Params {
    batch_size: u32,
    channels: u32,
    out_height: u32,
    out_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    num_blocks_h: u32,  // Number of blocks in height
    num_blocks_w: u32,  // Number of blocks in width
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;  // [B, C*K_h*K_w, L]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [B, C, H, W]

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

    var sum = 0.0;

    // Accumulate from all blocks that overlap this output position
    for (var block_h = 0u; block_h < params.num_blocks_h; block_h++) {
        for (var block_w = 0u; block_w < params.num_blocks_w; block_w++) {
            let block_idx = block_h * params.num_blocks_w + block_w;

            // Check if this output position overlaps with this block
            for (var kh = 0u; kh < params.kernel_height; kh++) {
                for (var kw = 0u; kw < params.kernel_width; kw++) {
                    let ih_base = block_h * params.stride;
                    let iw_base = block_w * params.stride;
                    let ih = ih_base + kh * params.dilation;
                    let iw = iw_base + kw * params.dilation;

                    // Check if this kernel position maps to current output position
                    if (ih >= params.padding && iw >= params.padding) {
                        let oh_from_block = ih - params.padding;
                        let ow_from_block = iw - params.padding;

                        if (oh_from_block == oh && ow_from_block == ow) {
                            // This block contributes to current output position
                            let channel_offset = (c * params.kernel_height + kh) * params.kernel_width + kw;
                            let in_idx = (b * (params.channels * params.kernel_height * params.kernel_width) + channel_offset)
                                          * (params.num_blocks_h * params.num_blocks_w) + block_idx;
                            sum += input[in_idx];
                        }
                    }
                }
            }
        }
    }

    let out_idx = ((b * params.channels + c) * params.out_height + oh) * params.out_width + ow;
    output[out_idx] = sum;
}
