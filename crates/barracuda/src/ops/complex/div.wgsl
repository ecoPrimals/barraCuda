// SPDX-License-Identifier: AGPL-3.0-or-later
// Complex Division: (a+bi)/(c+di) = (a+bi)(c-di)/(c²+d²)

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params { num_complex: u32, }

fn complex_mul(z1: vec2<f32>, z2: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z1.x * z2.x - z1.y * z2.y, z1.x * z2.y + z1.y * z2.x);
}

fn complex_div(z1: vec2<f32>, z2: vec2<f32>) -> vec2<f32> {
    let denom = dot(z2, z2);  // c²+d²
    let conj_z2 = vec2<f32>(z2.x, -z2.y);
    let num = complex_mul(z1, conj_z2);
    return num / denom;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_complex) { return; }
    let base = idx * 2u;
    let z1 = vec2<f32>(input_a[base], input_a[base + 1u]);
    let z2 = vec2<f32>(input_b[base], input_b[base + 1u]);
    let result = complex_div(z1, z2);
    output[base] = result.x;
    output[base + 1u] = result.y;
}
