// IFFT Normalization Shader
// Divides each complex number by N (the FFT size)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: NormalizeParams;

struct NormalizeParams {
    degree: u32,
    _padding0: u32,
    _padding1: u32,
    _padding2: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.degree) {
        return;
    }
    
    let scale = 1.0 / f32(params.degree);
    let base = idx * 2u;
    
    // Load complex number
    let real = input[base];
    let imag = input[base + 1u];
    
    // Normalize by 1/N
    output[base] = real * scale;
    output[base + 1u] = imag * scale;
}
