// SPDX-License-Identifier: AGPL-3.0-only
// Instance Normalization - Normalize per instance (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    batch: u32,
    channels: u32,
    spatial_size: u32,
    epsilon: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let instance_idx = global_id.x;
    let total_instances = params.batch * params.channels;
    
    if (instance_idx >= total_instances) {
        return;
    }
    
    let base_idx = instance_idx * params.spatial_size;
    
    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < params.spatial_size; i = i + 1u) {
        sum = sum + input[base_idx + i];
    }
    let mean = sum / f64(params.spatial_size);
    
    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < params.spatial_size; i = i + 1u) {
        let diff = input[base_idx + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f64(params.spatial_size);
    
    // Normalize
    let std_dev = sqrt_f64(variance + params.epsilon);
    for (var i = 0u; i < params.spatial_size; i = i + 1u) {
        let idx = base_idx + i;
        output[idx] = (input[idx] - mean) / std_dev;
    }
}
