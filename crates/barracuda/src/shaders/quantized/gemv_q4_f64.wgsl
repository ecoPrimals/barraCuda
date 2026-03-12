// SPDX-License-Identifier: AGPL-3.0-only
// Q4_0 GEMV (General Matrix-Vector Multiply) Shader (f64 canonical)
//
// Computes y = A @ x where A is Q4_0 quantized.
// Performs on-the-fly dequantization during computation.
//
// This is more efficient than dequant-then-multiply for inference:
//   - Reduces memory bandwidth (read 4-bit, not 32-bit)
//   - Uses GPU's parallelism for dequantization
//   - Ideal for LLM inference where weight access dominates

struct GemvParams {
    m: u32,
    k: u32,
    k_blocks: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a_quant: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f64>;
@group(0) @binding(2) var<storage, read_write> y: array<f64>;
@group(0) @binding(3) var<uniform> params: GemvParams;

const BLOCK_SIZE: u32 = 32u;
const BYTES_PER_BLOCK: u32 = 18u;
const WORKGROUP_SIZE: u32 = 256u;

var<workgroup> shared_x: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>
) {
    let row = global_id.x;

    if (row >= params.m) {
        return;
    }

    var sum: f64 = 0.0;

    let num_chunks = (params.k + WORKGROUP_SIZE - 1u) / WORKGROUP_SIZE;

    for (var chunk = 0u; chunk < num_chunks; chunk = chunk + 1u) {
        let x_idx = chunk * WORKGROUP_SIZE + local_id.x;
        if (x_idx < params.k) {
            shared_x[local_id.x] = x[x_idx];
        } else {
            shared_x[local_id.x] = 0.0;
        }
        workgroupBarrier();

        let start_block = chunk * (WORKGROUP_SIZE / BLOCK_SIZE);
        let end_block = min(start_block + (WORKGROUP_SIZE / BLOCK_SIZE), params.k_blocks);

        for (var block = start_block; block < end_block; block = block + 1u) {
            let byte_offset = (row * params.k_blocks + block) * BYTES_PER_BLOCK;
            let u32_offset = byte_offset / 4u;

            let raw0 = a_quant[u32_offset];
            let scale = f64(unpack2x16float(raw0 & 0xFFFFu).x);

            let data_u32_offset = u32_offset;
            let d0_raw = a_quant[data_u32_offset] >> 16u | (a_quant[data_u32_offset + 1u] << 16u);
            let d1_raw = a_quant[data_u32_offset + 1u] >> 16u | (a_quant[data_u32_offset + 2u] << 16u);
            let d2_raw = a_quant[data_u32_offset + 2u] >> 16u | (a_quant[data_u32_offset + 3u] << 16u);
            let d3_raw = a_quant[data_u32_offset + 3u] >> 16u | (a_quant[data_u32_offset + 4u] << 16u);

            let block_start = block * BLOCK_SIZE;
            let local_start = block_start - chunk * WORKGROUP_SIZE;

            sum = sum + dequant_dot_q4(d0_raw, scale, &shared_x, local_start, 0u);
            sum = sum + dequant_dot_q4(d1_raw, scale, &shared_x, local_start, 8u);
            sum = sum + dequant_dot_q4(d2_raw, scale, &shared_x, local_start, 16u);
            sum = sum + dequant_dot_q4(d3_raw, scale, &shared_x, local_start, 24u);
        }

        workgroupBarrier();
    }

    y[row] = sum;
}

fn dequant_dot_q4(packed: u32, scale: f64, x_shared: ptr<workgroup, array<f64, 256>>, base: u32, offset: u32) -> f64 {
    var sum: f64 = 0.0;

    for (var i = 0u; i < 8u; i = i + 1u) {
        let nibble = (packed >> (i * 4u)) & 0xFu;
        let q = i32(nibble) - 8;
        let val = scale * f64(q);
        let x_idx = base + offset + i;
        if (x_idx < 256u) {
            sum = sum + val * (*x_shared)[x_idx];
        }
    }

    return sum;
}
