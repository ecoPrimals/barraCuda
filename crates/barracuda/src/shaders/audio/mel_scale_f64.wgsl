// MelScale - Mel filterbank for audio feature extraction
// Converts linear frequency scale to mel scale
// Input: Power spectrogram [n_frames, n_freqs]
// Output: Mel spectrogram [n_frames, n_mels]

struct Params {
    n_frames: u32,
    n_freqs: u32,
    n_mels: u32,
    sample_rate: f32,
    f_min: f32,
    f_max: f32,
}

@group(0) @binding(0) var<storage, read> spectrogram: array<f64>;  // [n_frames, n_freqs]
@group(0) @binding(1) var<storage, read> filterbank: array<f64>;   // [n_mels, n_freqs]
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // [n_frames, n_mels]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let frame_idx = global_id.x;
    let mel_idx = global_id.y;
    
    if (frame_idx >= params.n_frames || mel_idx >= params.n_mels) {
        return;
    }
    
    var mel_spec: f64 = 0.0;
    
    // Apply filterbank to spectrogram
    for (var f: u32 = 0u; f < params.n_freqs; f = f + 1u) {
        let spec_idx = frame_idx * params.n_freqs + f;
        let filter_idx = mel_idx * params.n_freqs + f;
        mel_spec = mel_spec + spectrogram[spec_idx] * filterbank[filter_idx];
    }
    
    let output_idx = frame_idx * params.n_mels + mel_idx;
    output[output_idx] = mel_spec;
}
