// WindowFunction - Various windowing functions for signal processing (f64 canonical)
// Implements Hann, Hamming, Blackman, Bartlett, and Rectangular windows

struct Params {
    length: u32,
    window_type: u32,  // 0=Hann, 1=Hamming, 2=Blackman, 3=Bartlett, 4=Rectangular
}

@group(0) @binding(0) var<storage, read_write> output: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.length) {
        return;
    }
    
    let n = f64(idx);
    let length_f = f64(params.length);
    let pi = f64(3.14159265358979323846);
    
    var val: f64;
    
    switch (params.window_type) {
        case 0u: { // Hann
            val = f64(0.5) * (f64(1.0) - cos_f64(f64(2.0) * pi * n / (length_f - f64(1.0))));
        }
        case 1u: { // Hamming
            val = f64(0.54) - f64(0.46) * cos_f64(f64(2.0) * pi * n / (length_f - f64(1.0)));
        }
        case 2u: { // Blackman
            val = f64(0.42) - f64(0.5) * cos_f64(f64(2.0) * pi * n / (length_f - f64(1.0))) +
                  f64(0.08) * cos_f64(f64(4.0) * pi * n / (length_f - f64(1.0)));
        }
        case 3u: { // Bartlett
            val = f64(1.0) - abs(f64(2.0) * n / (length_f - f64(1.0)) - f64(1.0));
        }
        case 4u: { // Rectangular
            val = f64(1.0);
        }
        default: {
            val = f64(1.0);
        }
    }
    
    output[idx] = val;
}
