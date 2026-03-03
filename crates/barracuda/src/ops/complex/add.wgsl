// Complex Addition Shader
//
// Operation: (a + bi) + (c + di) = (a+c) + (b+d)i
//
// Architecture:
// - Complex stored as vec2<f32> (real, imag)
// - Native vec2 addition (SIMD-optimized)
// - Workgroup size 256 for optimal GPU occupancy
//
// Performance: ~1 GPU cycle per operation (trivial)

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    num_complex: u32,  // Number of complex numbers (not f32s)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_complex) {
        return;
    }
    
    // Load complex numbers as vec2
    let base = idx * 2u;
    let z1 = vec2<f32>(input_a[base], input_a[base + 1u]);
    let z2 = vec2<f32>(input_b[base], input_b[base + 1u]);
    
    // Complex addition (native vec2 operation!)
    let result = z1 + z2;
    
    // Store result
    output[base] = result.x;      // Real part
    output[base + 1u] = result.y; // Imaginary part
}
