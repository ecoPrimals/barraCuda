// SPDX-License-Identifier: AGPL-3.0-or-later
// Group Normalization - Normalize within groups (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    batch: u32,
    channels: u32,
    num_groups: u32,
    group_size: u32,
    spatial_size: u32,
    epsilon: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let group_idx = global_id.x;
    let total_groups = params.batch * params.num_groups;
    
    if (group_idx >= total_groups) {
        return;
    }
    
    let batch_idx = group_idx / params.num_groups;
    let group_in_batch = group_idx % params.num_groups;
    
    let group_start = batch_idx * params.channels * params.spatial_size +
                      group_in_batch * params.group_size * params.spatial_size;
    let group_elements = params.group_size * params.spatial_size;
    
    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < group_elements; i = i + 1u) {
        sum = sum + input[group_start + i];
    }
    let mean = sum / f64(group_elements);
    
    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < group_elements; i = i + 1u) {
        let diff = input[group_start + i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f64(group_elements);
    
    // Normalize
    let std_dev = sqrt_f64(variance + params.epsilon);
    for (var i = 0u; i < group_elements; i = i + 1u) {
        let idx = group_start + i;
        output[idx] = (input[idx] - mean) / std_dev;
    }
}
