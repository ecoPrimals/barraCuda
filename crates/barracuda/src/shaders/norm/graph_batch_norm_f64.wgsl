// SPDX-License-Identifier: AGPL-3.0-only
// Graph Batch Normalization - Batch normalization adapted for graph data (f64 canonical)
// Normalizes node features across the batch and feature dimensions
// Similar to standard batch norm, but operates on graph nodes
//
// Algorithm:
// 1. Compute mean: μ = (1/N) Σ x_i
// 2. Compute variance: σ^2 = (1/N) Σ (x_i - μ)^2
// 3. Normalize: x_norm = (x - μ) / sqrt(σ^2 + ε)
// 4. Scale and shift: y = γ * x_norm + β

struct Params {
    num_nodes: u32,
    num_features: u32,
    epsilon: f64,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;      // [num_nodes, num_features]
@group(0) @binding(2) var<storage, read> gamma: array<f64>;      // [num_features] (scale)
@group(0) @binding(3) var<storage, read> beta: array<f64>;       // [num_features] (shift)
@group(0) @binding(4) var<storage, read_write> output: array<f64>; // [num_nodes, num_features]
@group(0) @binding(5) var<storage, read_write> mean: array<f64>;    // [num_features] (computed)
@group(0) @binding(6) var<storage, read_write> variance: array<f64>; // [num_features] (computed)

// Step 1: Compute mean per feature
@compute @workgroup_size(256)
fn compute_mean(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let feature_idx = global_id.x;
    if (feature_idx >= params.num_features) {
        return;
    }

    var sum = 0.0;
    for (var n = 0u; n < params.num_nodes; n++) {
        sum += input[n * params.num_features + feature_idx];
    }
    mean[feature_idx] = sum / f64(params.num_nodes);
}

// Step 2: Compute variance per feature
@compute @workgroup_size(256)
fn compute_variance(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let feature_idx = global_id.x;
    if (feature_idx >= params.num_features) {
        return;
    }

    let mu = mean[feature_idx];
    var sum_sq = 0.0;
    for (var n = 0u; n < params.num_nodes; n++) {
        let diff = input[n * params.num_features + feature_idx] - mu;
        sum_sq += diff * diff;
    }
    variance[feature_idx] = sum_sq / f64(params.num_nodes);
}

// Step 3: Normalize, scale, and shift
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.num_nodes * params.num_features;
    
    if (idx >= total) {
        return;
    }

    let node = idx / params.num_features;
    let feature = idx % params.num_features;

    let x = input[idx];
    let mu = mean[feature];
    let var_val = variance[feature];
    
    // Normalize
    let x_norm = (x - mu) / sqrt_f64(var_val + params.epsilon);
    
    // Scale and shift
    output[idx] = gamma[feature] * x_norm + beta[feature];
}
