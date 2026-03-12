// SPDX-License-Identifier: AGPL-3.0-only
// TimeStretch - Time-domain stretching without pitch change
// Phase vocoder-based time stretching
// Input: Signal [length]
// Output: Stretched signal [output_length]

struct Params {
    input_length: u32,
    output_length: u32,
    n_fft: u32,
    hop_length: u32,
    stretched_hop: u32,
    num_frames: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> window: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<storage, read_write> window_sum: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let frame_idx = global_id.x;
    
    if (frame_idx >= params.num_frames) {
        return;
    }
    
    let in_pos = frame_idx * params.hop_length;
    let out_pos = frame_idx * params.stretched_hop;
    
    if (out_pos + params.n_fft > params.output_length) {
        return;
    }
    
    // Copy frame with phase adjustment (simplified - full implementation would use phase vocoder)
    for (var n: u32 = 0u; n < params.n_fft; n = n + 1u) {
        if (in_pos + n < params.input_length && out_pos + n < params.output_length) {
            let windowed = input[in_pos + n] * window[n];
            output[out_pos + n] = output[out_pos + n] + windowed;
            window_sum[out_pos + n] = window_sum[out_pos + n] + window[n] * window[n];
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
    
    if (window_sum[idx] > 1e-8) {
        output[idx] = output[idx] / window_sum[idx];
    }
}
