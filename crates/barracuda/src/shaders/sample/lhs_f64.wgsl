// SPDX-License-Identifier: AGPL-3.0-only
// Latin Hypercube Sampling (LHS) - GPU accelerated (f64 canonical)
// Each thread generates one sample point, stratified across dimensions
//
// Input: bounds array [lo0, hi0, lo1, hi1, ...] interleaved
// Output: samples [x0_d0, x0_d1, ..., x1_d0, x1_d1, ...]
//
// Applications: design of experiments, surrogate model training, hyperparameter search
// Reference: McKay et al. (1979), "A comparison of three methods..."

@group(0) @binding(0) var<storage, read> bounds: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<storage, read_write> permutations: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n_samples: u32,
    n_dims: u32,
    seed: u32,
    _pad: u32,
}

// PCG random number generator (stateless, thread-safe)
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Generate uniform float in [0, 1) from seed
fn random_float(seed: u32) -> f64 {
    return f64(pcg_hash(seed)) / 4294967296.0;
}

// Generate random float for sample i, dimension d, component c
fn rand_for_sample(sample_idx: u32, dim: u32, component: u32, base_seed: u32) -> f64 {
    let seed = base_seed + sample_idx * 1000u + dim * 100u + component;
    return random_float(seed);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sample_idx = global_id.x;
    if (sample_idx >= params.n_samples) {
        return;
    }

    let n_samples = params.n_samples;
    let n_dims = params.n_dims;
    let interval_width = 1.0 / f64(n_samples);

    // For each dimension, compute the stratified sample
    for (var d = 0u; d < n_dims; d = d + 1u) {
        // Get the permuted interval index for this sample in this dimension
        let perm_idx = d * n_samples + sample_idx;
        let interval_idx = permutations[perm_idx];

        // Get bounds for this dimension
        let lo = bounds[d * 2u];
        let hi = bounds[d * 2u + 1u];
        let dim_range = hi - lo;

        // Compute interval in [0, 1] space
        let interval_lo = f64(interval_idx) * interval_width;

        // Random offset within interval
        let offset = rand_for_sample(sample_idx, d, 0u, params.seed);

        // Map to actual bounds
        let normalized = interval_lo + offset * interval_width;
        let value = lo + normalized * dim_range;

        // Store result
        output[sample_idx * n_dims + d] = value;
    }
}
