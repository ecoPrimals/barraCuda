// SPDX-License-Identifier: AGPL-3.0-only
// Q4_0 Dequantization Shader (f64 canonical)
//
// Dequantizes 4-bit quantized weights to f64 for inference.
// Each block contains 32 elements packed in 18 bytes:
//   - 2 bytes: scale factor (f16)
//   - 16 bytes: 32 x 4-bit quantized values
//
// Deep Debt Compliance:
// - Pure WGSL (no vendor extensions)
// - Workgroup size 256 (universal)
// - Coalesced memory access patterns

struct DequantParams {
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: DequantParams;

const BLOCK_SIZE: u32 = 32u;
const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_idx = global_id.x;

    if (block_idx >= params.num_blocks) {
        return;
    }

    let byte_offset = block_idx * 18u;
    let u32_offset = byte_offset / 4u;

    let raw0 = input[u32_offset];
    let raw1 = input[u32_offset + 1u];
    let raw2 = input[u32_offset + 2u];
    let raw3 = input[u32_offset + 3u];
    let raw4 = input[u32_offset + 4u];

    let byte_mod = byte_offset % 4u;
    var scale_bits: u32;
    if (byte_mod == 0u) {
        scale_bits = raw0 & 0xFFFFu;
    } else if (byte_mod == 2u) {
        scale_bits = raw0 >> 16u;
    } else {
        scale_bits = ((raw0 >> (byte_mod * 8u)) | (raw1 << (32u - byte_mod * 8u))) & 0xFFFFu;
    }

    let scale = f64(unpack2x16float(scale_bits).x);

    let out_base = block_idx * BLOCK_SIZE;

    let data_byte_offset = byte_offset + 2u;
    let data_u32_offset = data_byte_offset / 4u;
    let data_byte_mod = data_byte_offset % 4u;

    let d0 = input[data_u32_offset];
    let d1 = input[data_u32_offset + 1u];
    let d2 = input[data_u32_offset + 2u];
    let d3 = input[data_u32_offset + 3u];
    let d4 = input[data_u32_offset + 4u];

    for (var i = 0u; i < 16u; i = i + 1u) {
        let byte_idx = i;
        let word_idx = byte_idx / 4u;
        let local_byte = byte_idx % 4u;

        var byte_val: u32;
        if (word_idx == 0u) {
            byte_val = (d0 >> (local_byte * 8u + data_byte_mod * 8u)) & 0xFFu;
        } else if (word_idx == 1u) {
            byte_val = (d1 >> (local_byte * 8u)) & 0xFFu;
        } else if (word_idx == 2u) {
            byte_val = (d2 >> (local_byte * 8u)) & 0xFFu;
        } else {
            byte_val = (d3 >> (local_byte * 8u)) & 0xFFu;
        }

        let q0 = i32(byte_val & 0xFu) - 8;
        let q1 = i32((byte_val >> 4u) & 0xFu) - 8;

        let idx0 = out_base + i * 2u;
        let idx1 = out_base + i * 2u + 1u;

        output[idx0] = scale * f64(q0);
        output[idx1] = scale * f64(q1);
    }
}
