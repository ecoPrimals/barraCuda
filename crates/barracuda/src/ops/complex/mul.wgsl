// SPDX-License-Identifier: AGPL-3.0-only
// Complex Multiplication Shader
//
// Operation: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//
// **CRITICAL FOR FFT**: This is the bottleneck operation in FFT butterfly computations
//
// Algorithm:
// - z1 = a + bi, z2 = c + di
// - real(result) = a*c - b*d
// - imag(result) = a*d + b*c
//
// Performance: 4 multiplications + 2 additions = ~2-3 GPU cycles
//
// Architecture:
// - Complex stored as vec2<f32> (real, imag)
// - Direct computation (no table lookups)
// - Workgroup size 256 for optimal GPU occupancy

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    num_complex: u32,  // Number of complex numbers
}

/// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
fn complex_mul(z1: vec2<f32>, z2: vec2<f32>) -> vec2<f32> {
    let a = z1.x;  // real(z1)
    let b = z1.y;  // imag(z1)
    let c = z2.x;  // real(z2)
    let d = z2.y;  // imag(z2)
    
    let re = a * c - b * d;  // Real part: ac - bd
    let im = a * d + b * c;  // Imaginary part: ad + bc
    
    return vec2<f32>(re, im);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_complex) {
        return;
    }
    
    // Load complex numbers
    let base = idx * 2u;
    let z1 = vec2<f32>(input_a[base], input_a[base + 1u]);
    let z2 = vec2<f32>(input_b[base], input_b[base + 1u]);
    
    // Complex multiplication
    let result = complex_mul(z1, z2);
    
    // Store result
    output[base] = result.x;      // Real part
    output[base + 1u] = result.y; // Imaginary part
}
