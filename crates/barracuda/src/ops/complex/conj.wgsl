// SPDX-License-Identifier: AGPL-3.0-or-later
// Complex Conjugate Shader
// Operation: conj(a + bi) = a - bi
// **CRITICAL FOR FFT**: Used in inverse FFT and normalization

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    num_complex: u32,
}

fn complex_conj(z: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(z.x, -z.y);  // Keep real, negate imaginary
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_complex) {
        return;
    }
    
    let base = idx * 2u;
    let z = vec2<f32>(input[base], input[base + 1u]);
    let result = complex_conj(z);
    
    output[base] = result.x;
    output[base + 1u] = result.y;
}
