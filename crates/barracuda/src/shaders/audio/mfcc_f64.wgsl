// MFCC - Mel-Frequency Cepstral Coefficients (f64 canonical)
// Extracts MFCC features from mel spectrogram
// Input: Mel spectrogram [n_frames, n_mels]
// Output: MFCC features [n_frames, n_mfcc]

struct Params {
    n_frames: u32,
    n_mels: u32,
    n_mfcc: u32,
}

@group(0) @binding(0) var<storage, read> mel_spectrogram: array<f64>; // [n_frames, n_mels]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;     // [n_frames, n_mfcc]
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let frame_idx = global_id.x;
    let k = global_id.y;  // MFCC coefficient index
    
    if (frame_idx >= params.n_frames || k >= params.n_mfcc) {
        return;
    }
    
    var sum: f64 = f64(0.0);
    let pi = f64(3.14159265358979323846);
    let n_mels_f = f64(params.n_mels);
    
    // Apply DCT-II to log mel spectrogram
    for (var n: u32 = 0u; n < params.n_mels; n = n + 1u) {
        let mel_idx = frame_idx * params.n_mels + n;
        let mel_val = mel_spectrogram[mel_idx];
        
        // Log compression (with epsilon for numerical stability)
        let log_mel = log_f64(mel_val + f64(1e-8));
        
        // DCT-II: cos(pi * k * (n + 0.5) / n_mels)
        let angle = pi * f64(k) * (f64(n) + f64(0.5)) / n_mels_f;
        sum = sum + log_mel * cos_f64(angle);
    }
    
    // DCT-II normalization
    let output_idx = frame_idx * params.n_mfcc + k;
    output[output_idx] = sum * sqrt_f64(f64(2.0) / n_mels_f);
}
