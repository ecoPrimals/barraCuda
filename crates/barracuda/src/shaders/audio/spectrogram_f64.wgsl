// SPDX-License-Identifier: AGPL-3.0-or-later
// Spectrogram - Power spectrogram computation (f64 canonical)
// Computes magnitude squared of STFT
// Input: Complex STFT [real, imag, real, imag, ...]
// Output: Power spectrogram [magnitude^power]

struct Params {
    size: u32,      // Number of complex pairs
    power: f64,     // 1.0 for magnitude, 2.0 for power
}

@group(0) @binding(0) var<storage, read> input: array<f64>;        // [real, imag, real, imag, ...]
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // [magnitude^power]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Read complex pair: real at 2*idx, imag at 2*idx+1
    let real = input[2u * idx];
    let imag = input[2u * idx + 1u];
    
    // Compute magnitude
    let magnitude = sqrt_f64(real * real + imag * imag);
    
    // Apply power
    output[idx] = pow_f64(magnitude, params.power);
}
