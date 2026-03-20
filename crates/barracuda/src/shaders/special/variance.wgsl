// SPDX-License-Identifier: AGPL-3.0-or-later
// Variance and standard deviation computation
// Var(X) = E[(X-μ)²] = E[X²] - E[X]²
//
// Input: vector of values
// Output: variance (or std dev based on mode)
//
// Applications: statistics, normalization, feature scaling
// Reference: Standard statistical formulas

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,        // Length of each vector
    num_vectors: u32, // Number of vectors to process
    stride: u32,      // Stride between vectors
    ddof: u32,        // Delta degrees of freedom (0=population, 1=sample)
    mode: u32,        // 0=variance, 1=std_dev
}

// Two-pass algorithm for numerical stability
fn compute_variance(
    data: ptr<storage, array<f32>, read>,
    offset: u32,
    size: u32,
    ddof: u32
) -> f32 {
    if (size <= ddof) {
        return 0.0;
    }

    // Pass 1: compute mean
    var sum: f32 = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        sum = sum + data[offset + i];
    }
    let mean = sum / f32(size);

    // Pass 2: compute variance
    var var_sum: f32 = 0.0;
    for (var i = 0u; i < size; i = i + 1u) {
        let d = data[offset + i] - mean;
        var_sum = var_sum + d * d;
    }

    return var_sum / f32(size - ddof);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_vectors) {
        return;
    }

    let offset = idx * params.stride;
    var result = compute_variance(&input, offset, params.size, params.ddof);

    if (params.mode == 1u) {
        result = sqrt(result);  // Standard deviation
    }

    output[idx] = result;
}
