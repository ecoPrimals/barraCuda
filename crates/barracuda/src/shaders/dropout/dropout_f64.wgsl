// Dropout - Random dropout for regularization (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)
//
// Note: Uses simple LCG for randomness (deterministic seed)

struct Params {
    size: u32,
    probability: f64,
    scale: f64,
    seed: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

// Linear Congruential Generator for pseudo-random numbers
fn lcg(seed: u32) -> u32 {
    return (1103515245u * seed + 12345u) & 0x7fffffffu;
}

fn random_f64(seed: u32) -> f64 {
    return f64(lcg(seed)) / f64(0x7fffffffu);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.size) {
        return;
    }
    
    // Generate random number for this element
    let seed = params.seed + idx;
    let rand = random_f64(seed);
    
    // Apply dropout: zero out with probability p, scale by 1/(1-p) otherwise
    if (rand < params.probability) {
        output[idx] = 0.0;
    } else {
        output[idx] = input[idx] * params.scale;
    }
}
