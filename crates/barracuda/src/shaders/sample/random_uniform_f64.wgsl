// Uniform random sampling - GPU accelerated (f64 canonical)
// Uses PCG hash for thread-safe parallel random generation
//
// Input: bounds array [lo0, hi0, lo1, hi1, ...] interleaved
// Output: samples [x0_d0, x0_d1, ..., x1_d0, x1_d1, ...]
//
// Applications: Monte Carlo, random initialization, baseline sampling

@group(0) @binding(0) var<storage, read> bounds: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    n_samples: u32,
    n_dims: u32,
    seed: u32,
    _pad: u32,
}

// PCG random number generator (stateless)
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate uniform float in [0, 1)
fn random_float(seed: u32) -> f64 {
    return f64(pcg_hash(seed)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    if (sample_idx >= params.n_samples) {
        return;
    }

    // Generate random point for each dimension
    for (var d = 0u; d < params.n_dims; d = d + 1u) {
        // Unique seed for each (sample, dimension) pair
        let seed = params.seed + sample_idx * 1000u + d;
        let rand = random_float(seed);

        // Get bounds for this dimension
        let lo = bounds[d * 2u];
        let hi = bounds[d * 2u + 1u];

        // Scale to bounds
        let value = lo + rand * (hi - lo);

        // Store result
        output[sample_idx * params.n_dims + d] = value;
    }
}
