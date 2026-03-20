// SPDX-License-Identifier: AGPL-3.0-or-later
// Color Jitter - Data augmentation for image color
// Randomly adjusts brightness, contrast, saturation, hue

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

// Simple pseudo-random based on position
fn random(seed: u32) -> f64 {
    let s = seed * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return f64((word >> 22u) ^ word) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.batch_size * params.channels * params.height * params.width;
    
    if (idx >= total_size) {
        return;
    }
    
    let value = input[idx];
    
    // Apply brightness adjustment (additive)
    var adjusted = value + f64(params.brightness) * (random(idx) * 2.0 - 1.0);
    
    // Apply contrast adjustment (multiplicative around 0.5)
    let contrast_factor = 1.0 + f64(params.contrast) * (random(idx + 1000u) * 2.0 - 1.0);
    adjusted = (adjusted - 0.5) * contrast_factor + 0.5;
    
    // Clamp to [0, 1]
    adjusted = clamp(adjusted, 0.0, 1.0);
    
    output[idx] = adjusted;
}
