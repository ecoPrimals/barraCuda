// SPDX-License-Identifier: AGPL-3.0-only
// ISTFT - Inverse Short-Time Fourier Transform (f64 canonical)
// Reconstructs time-domain signal from STFT using overlap-add method
// Input: Complex STFT [num_frames, bins_per_frame] as [real, imag, real, imag, ...]
// Output: Time-domain signal [output_length]

struct Params {
    num_frames: u32,
    n_fft: u32,
    hop_length: u32,
    bins_per_frame: u32,
    output_length: u32,
}

@group(0) @binding(0) var<storage, read> stft_data: array<f64>;      // [real, imag, real, imag, ...]
@group(0) @binding(1) var<storage, read> window: array<f64>;          // [n_fft]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // [output_length]
@group(0) @binding(3) var<storage, read_write> window_sum: array<f64>; // [output_length]
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let frame_idx = global_id.x;
    
    if (frame_idx >= params.num_frames) {
        return;
    }
    
    let start = frame_idx * params.hop_length;
    let pi = f64(3.14159265358979323846);
    
    // Inverse DFT for this frame
    for (var n: u32 = 0u; n < params.n_fft; n = n + 1u) {
        var frame_val: f64 = f64(0.0);
        
        for (var k: u32 = 0u; k < params.bins_per_frame; k = k + 1u) {
            let stft_idx = (frame_idx * params.bins_per_frame + k) * 2u;
            let real = stft_data[stft_idx];
            let imag = stft_data[stft_idx + 1u];
            
            let angle = f64(2.0) * pi * f64(k) * f64(n) / f64(params.n_fft);
            frame_val = frame_val + real * cos_f64(angle) - imag * sin_f64(angle);
        }
        
        frame_val = frame_val / f64(params.n_fft);
        
        // Overlap-add with window
        let output_idx = start + n;
        if (output_idx < params.output_length) {
            let windowed = frame_val * window[n];
            output[output_idx] = output[output_idx] + windowed;
            window_sum[output_idx] = window_sum[output_idx] + window[n] * window[n];
        }
    }
}

// Normalization pass
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.output_length) {
        return;
    }
    
    if (window_sum[idx] > f64(1e-8)) {
        output[idx] = output[idx] / window_sum[idx];
    }
}
