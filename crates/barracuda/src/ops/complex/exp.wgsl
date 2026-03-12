// SPDX-License-Identifier: AGPL-3.0-only
// Complex Exponential Shader
// Operation: exp(a + bi) = exp(a)[cos(b) + i·sin(b)]
// **CRITICAL FOR FFT**: Generates twiddle factors W_N^k = exp(-2πik/N)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    num_complex: u32,
}

/// Euler's formula: exp(a+bi) = exp(a)[cos(b) + i·sin(b)]
fn complex_exp(z: vec2<f32>) -> vec2<f32> {
    let exp_re = exp(z.x);        // exp(a)
    let cos_im = cos(z.y);        // cos(b)
    let sin_im = sin(z.y);        // sin(b)
    
    return vec2<f32>(
        exp_re * cos_im,          // exp(a)·cos(b)
        exp_re * sin_im           // exp(a)·sin(b)
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_complex) {
        return;
    }
    
    let base = idx * 2u;
    let z = vec2<f32>(input[base], input[base + 1u]);
    let result = complex_exp(z);
    
    output[base] = result.x;
    output[base + 1u] = result.y;
}
