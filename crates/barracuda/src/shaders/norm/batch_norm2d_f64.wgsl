// SPDX-License-Identifier: AGPL-3.0-or-later
// Batch Norm 2D - Batch normalization for CNNs (complete implementation) (f64 canonical)
// Normalizes activations across batch dimension
//
// Algorithm:
// 1. Compute mean per channel: μ_c = (1/NHW) Σ x
// 2. Compute variance per channel: σ²_c = (1/NHW) Σ (x - μ)²
// 3. Normalize: x_norm = (x - μ) / sqrt(σ² + ε)
// 4. Scale and shift: y = γ * x_norm + β

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    epsilon: f64,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;          // [B, C, H, W]
@group(0) @binding(2) var<storage, read> gamma: array<f64>;          // [C] (scale)
@group(0) @binding(3) var<storage, read> beta: array<f64>;           // [C] (shift)
@group(0) @binding(4) var<storage, read_write> mean: array<f64>;     // [C] (computed)
@group(0) @binding(5) var<storage, read_write> variance: array<f64>; // [C] (computed)
@group(0) @binding(6) var<storage, read_write> output: array<f64>;   // [B, C, H, W]

// Step 1: Compute mean per channel
@compute @workgroup_size(256)
fn compute_mean(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;
    if (c >= params.channels) {
        return;
    }

    var sum = f64(0.0);
    let spatial_size = params.height * params.width;

    for (var b = 0u; b < params.batch_size; b++) {
        for (var h = 0u; h < params.height; h++) {
            for (var w = 0u; w < params.width; w++) {
                let idx = ((b * params.channels + c) * params.height + h) * params.width + w;
                sum += input[idx];
            }
        }
    }

    mean[c] = sum / f64(params.batch_size * spatial_size);
}

// Step 2: Compute variance per channel
@compute @workgroup_size(256)
fn compute_variance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;
    if (c >= params.channels) {
        return;
    }

    let mu = mean[c];
    var sum_sq = f64(0.0);
    let spatial_size = params.height * params.width;

    for (var b = 0u; b < params.batch_size; b++) {
        for (var h = 0u; h < params.height; h++) {
            for (var w = 0u; w < params.width; w++) {
                let idx = ((b * params.channels + c) * params.height + h) * params.width + w;
                let diff = input[idx] - mu;
                sum_sq += diff * diff;
            }
        }
    }

    variance[c] = sum_sq / f64(params.batch_size * spatial_size);
}

// Step 3: Normalize, scale, and shift
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.batch_size * params.channels * params.height * params.width;

    if (idx >= total) {
        return;
    }

    // Decompose index
    let temp1 = idx / params.width;
    let w = idx % params.width;
    let temp2 = temp1 / params.height;
    let h = temp1 % params.height;
    let temp3 = temp2 / params.channels;
    let c = temp2 % params.channels;
    let b = temp3;

    let x = input[idx];
    let mu = mean[c];
    let var = variance[c];

    // Normalize
    let x_norm = (x - mu) / sqrt_f64(var + params.epsilon);

    // Scale and shift
    output[idx] = gamma[c] * x_norm + beta[c];
}
