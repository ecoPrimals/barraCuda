// SPDX-License-Identifier: AGPL-3.0-or-later
// z^n via De Moivre
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
struct Params { num_complex: u32, exponent: f32, }

fn complex_pow(z: vec2<f32>, n: f32) -> vec2<f32> {
    let r = length(z);
    let theta = atan2(z.y, z.x);
    let r_pow_n = pow(r, n);
    let n_theta = n * theta;
    return vec2<f32>(r_pow_n * cos(n_theta), r_pow_n * sin(n_theta));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_complex) { return; }
    let base = idx * 2u;
    let z = vec2<f32>(input[base], input[base + 1u]);
    let result = complex_pow(z, params.exponent);
    output[base] = result.x;
    output[base + 1u] = result.y;
}
