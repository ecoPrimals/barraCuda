// SPDX-License-Identifier: AGPL-3.0-only
// quantize_params_f64.wgsl — Affine quantization parameters via max-abs reduction (f64)
//
// **Math**: For symmetric affine quantization (scale, zero_point):
//   abs_max = max_i |data[i]|
//   scale = 127.0 / abs_max
//   zero_point = 0
//
// This shader computes the first-stage partial max: each thread reduces a block of
// `block_size` elements to one partial_max value. Host reduces partial_max to get
// final abs_max, then computes scale = 127.0 / abs_max.
//
// **Algorithm**: Thread i handles block [i*block_size, (i+1)*block_size), computes
// max(abs(data[j])) over j in that range, writes to partial_max[i].
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: data       array<vec2<u32>>  read       — input values
//   1: partial_max array<vec2<u32>> read_write — one max per block (for host reduction)
//   2: params     uniform
//
// Params: { n: u32, block_size: u32 }
//
// Dispatch: ceil((n + block_size - 1) / block_size / 256) workgroups
// Applications: INT8 quantization, model compression, inference acceleration.

@group(0) @binding(0) var<storage, read> data: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> partial_max: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    n: u32,
    block_size: u32,
    _pad2: u32,
    _pad3: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_idx = global_id.x;
    let n = params.n;
    let block_size = params.block_size;

    let start = block_idx * block_size;
    if (start >= n) {
        return;
    }

    let end = min(start + block_size, n);

    var m = f64(0.0);
    for (var j = start; j < end; j = j + 1u) {
        let v = bitcast<f64>(data[j]);
        let a = abs(v);
        if (a > m) {
            m = a;
        }
    }

    partial_max[block_idx] = bitcast<vec2<u32>>(m);
}
