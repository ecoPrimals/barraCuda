// SPDX-License-Identifier: AGPL-3.0-or-later
// Complex Absolute Value (Magnitude) Shader
// Operation: |a + bi| = sqrt(a² + b²)
// Uses native WGSL length() function

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    num_complex: u32,
}

fn complex_abs(z: vec2<f32>) -> f32 {
    return length(z);  // Native WGSL function: sqrt(x²+y²)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_complex) {
        return;
    }
    
    let base = idx * 2u;
    let z = vec2<f32>(input[base], input[base + 1u]);
    output[idx] = complex_abs(z);  // Single real output per complex input
}
