// SPDX-License-Identifier: AGPL-3.0-or-later
// GroupNorm: Group normalization (modern alternative to BatchNorm) (f64 canonical)
// CUDA equivalent: Custom kernels or PyTorch's GroupNorm
// Formula: output = (input - group_mean) / sqrt(group_var + epsilon) * gamma + beta
// Use cases: Small batch training, style transfer, generative models

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;  // Scale parameter (per channel)
@group(0) @binding(2) var<storage, read> beta: array<f64>;   // Shift parameter (per channel)
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<storage, read_write> stats: array<f64>;  // Group statistics (mean, var)

struct Params {
    batch_size: u32,
    channels: u32,
    spatial_size: u32,  // H * W
    num_groups: u32,
    channels_per_group: u32,
    epsilon: f64,
}
@group(0) @binding(5) var<uniform> params: Params;

// Shared memory for workgroup reductions
var<workgroup> shared_sum: array<f64, 256>;
var<workgroup> shared_sum_sq: array<f64, 256>;

// Pass 1: Compute group statistics (mean and variance)
@compute @workgroup_size(256)
fn compute_stats(@builtin(global_invocation_id) global_id: vec3<u32>,
                 @builtin(local_invocation_id) local_id: vec3<u32>,
                 @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let gid = global_id.x;
    let local_idx = local_id.x;
    let batch_idx = workgroup_id.z / params.num_groups;
    let group_idx = workgroup_id.z % params.num_groups;
    
    let group_size = params.channels_per_group * params.spatial_size;
    let group_start = batch_idx * params.channels * params.spatial_size + 
                      group_idx * params.channels_per_group * params.spatial_size;
    
    // Each thread accumulates a portion of the group
    var local_sum = 0.0;
    var local_sum_sq = 0.0;
    var count = 0u;
    
    for (var i = gid; i < group_size; i += 256u) {
        let idx = group_start + i;
        if (idx < arrayLength(&input)) {
            let val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
            count += 1u;
        }
    }
    
    shared_sum[local_idx] = local_sum;
    shared_sum_sq[local_idx] = local_sum_sq;
    workgroupBarrier();
    
    // Tree reduction within workgroup
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (local_idx < stride) {
            shared_sum[local_idx] += shared_sum[local_idx + stride];
            shared_sum_sq[local_idx] += shared_sum_sq[local_idx + stride];
        }
        workgroupBarrier();
    }
    
    // First thread computes mean and variance for this group
    if (local_idx == 0u) {
        let total = shared_sum[0];
        let total_sq = shared_sum_sq[0];
        let n = f64(group_size);
        
        let mean = total / n;
        let variance = (total_sq / n) - (mean * mean);
        
        // Store statistics: [mean, variance] for each group
        let stats_idx = workgroup_id.z * 2u;
        stats[stats_idx] = mean;
        stats[stats_idx + 1u] = variance;
    }
}

// Pass 2: Normalize using group statistics
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    
    if (gid >= params.batch_size * params.channels * params.spatial_size) {
        return;
    }
    
    // Determine which batch, channel, and position
    let spatial_idx = gid % params.spatial_size;
    let channel_idx = (gid / params.spatial_size) % params.channels;
    let batch_idx = gid / (params.channels * params.spatial_size);
    
    // Determine which group this channel belongs to
    let group_idx = channel_idx / params.channels_per_group;
    
    // Get group statistics
    let stats_base = (batch_idx * params.num_groups + group_idx) * 2u;
    let mean = stats[stats_base];
    let variance = stats[stats_base + 1u];
    
    // Normalize: (x - mean) / sqrt(variance + epsilon)
    let normalized = (input[gid] - mean) / sqrt_f64(variance + params.epsilon);
    
    // Apply affine transformation: gamma * normalized + beta
    output[gid] = gamma[channel_idx] * normalized + beta[channel_idx];
}
