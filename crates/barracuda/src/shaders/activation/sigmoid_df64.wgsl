// SPDX-License-Identifier: AGPL-3.0-only
// Sigmoid activation in DF64 precision — numerically stable
// sigmoid(x) = 1 / (1 + exp(-x))  for x >= 0
//            = exp(x) / (1 + exp(x))  for x < 0
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)

@group(0) @binding(0) var<storage, read> input_hi: array<f32>;
@group(0) @binding(1) var<storage, read> input_lo: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_hi: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_lo: array<f32>;
@group(0) @binding(4) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= size) { return; }

    let x = Df64(input_hi[idx], input_lo[idx]);
    let one = df64_from_f32(1.0);

    var result: Df64;
    if (x.hi >= 0.0) {
        let z = exp_df64(df64_neg(x));
        result = df64_div(one, df64_add(one, z));
    } else {
        let z = exp_df64(x);
        result = df64_div(z, df64_add(one, z));
    }

    output_hi[idx] = result.hi;
    output_lo[idx] = result.lo;
}
