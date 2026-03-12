// SPDX-License-Identifier: AGPL-3.0-only
// Layer normalization in DF64 precision
// y_i = gamma * (x_i - mean) / sqrt(var + eps) + beta
// Requires: df64_core.wgsl, df64_transcendentals.wgsl (via compile_shader_df64)
//
// Two-pass: pass 1 computes mean/variance, pass 2 normalizes.
// This shader handles pass 2 (normalize + affine).
// Mean and variance are precomputed and passed as uniforms.

struct LayerNormParams {
    size: u32,
    mean_hi: f32,
    mean_lo: f32,
    inv_std_hi: f32,
    inv_std_lo: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_hi: array<f32>;
@group(0) @binding(1) var<storage, read> input_lo: array<f32>;
@group(0) @binding(2) var<storage, read> gamma_hi: array<f32>;
@group(0) @binding(3) var<storage, read> gamma_lo: array<f32>;
@group(0) @binding(4) var<storage, read> beta_hi: array<f32>;
@group(0) @binding(5) var<storage, read> beta_lo: array<f32>;
@group(0) @binding(6) var<storage, read_write> output_hi: array<f32>;
@group(0) @binding(7) var<storage, read_write> output_lo: array<f32>;
@group(0) @binding(8) var<uniform> params: LayerNormParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }

    let x = Df64(input_hi[idx], input_lo[idx]);
    let mean = Df64(params.mean_hi, params.mean_lo);
    let inv_std = Df64(params.inv_std_hi, params.inv_std_lo);
    let g = Df64(gamma_hi[idx], gamma_lo[idx]);
    let b = Df64(beta_hi[idx], beta_lo[idx]);

    let centered = df64_sub(x, mean);
    let normalized = df64_mul(centered, inv_std);
    let result = df64_add(df64_mul(g, normalized), b);

    output_hi[idx] = result.hi;
    output_lo[idx] = result.lo;
}
