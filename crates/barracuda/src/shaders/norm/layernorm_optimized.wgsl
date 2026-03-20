// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optimized LayerNorm - 2-Pass Algorithm
//!
//! Performance Target: 10x improvement over 3-pass
//! Key Optimizations:
//! 1. Reduced from 3 passes to 2 passes
//! 2. Fused finalization + normalization
//! 3. Better shared memory usage
//! 4. Optimized workgroup size (128 threads)

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> gamma: array<f32>;
@group(0) @binding(2) var<storage, read> beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<storage, read_write> stats: array<f32>;  // Partial sums

struct Params {
    size: u32,
    epsilon: f32,
    _pad: u32,
    _pad2: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

// Shared memory for reductions (128 threads × 2 values)
var<workgroup> shared_sum: array<f32, 128>;
var<workgroup> shared_sum_sq: array<f32, 128>;

// Pass 1: Compute partial statistics per workgroup
@compute @workgroup_size(128)
fn compute_partial_stats(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    let wg_size = 128u;
    
    // Each thread accumulates multiple elements (coalesced access)
    var local_sum = 0.0;
    var local_sum_sq = 0.0;
    
    // Grid-stride loop for better occupancy
    var idx = gid;
    while (idx < params.size) {
        let value = input[idx];
        local_sum += value;
        local_sum_sq += value * value;
        idx += wg_size * gridDim().x;
    }
    
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    workgroupBarrier();
    
    // Optimized tree reduction (unrolled)
    if (wg_size >= 128u && tid < 64u) {
        shared_sum[tid] += shared_sum[tid + 64u];
        shared_sum_sq[tid] += shared_sum_sq[tid + 64u];
    }
    workgroupBarrier();
    
    if (tid < 32u) {
        // Warp-level reduction (no sync needed within warp on most GPUs)
        shared_sum[tid] += shared_sum[tid + 32u];
        shared_sum_sq[tid] += shared_sum_sq[tid + 32u];
        workgroupBarrier();
        
        if (tid < 16u) {
            shared_sum[tid] += shared_sum[tid + 16u];
            shared_sum_sq[tid] += shared_sum_sq[tid + 16u];
        }
        workgroupBarrier();
        
        if (tid < 8u) {
            shared_sum[tid] += shared_sum[tid + 8u];
            shared_sum_sq[tid] += shared_sum_sq[tid + 8u];
        }
        workgroupBarrier();
        
        if (tid < 4u) {
            shared_sum[tid] += shared_sum[tid + 4u];
            shared_sum_sq[tid] += shared_sum_sq[tid + 4u];
        }
        workgroupBarrier();
        
        if (tid < 2u) {
            shared_sum[tid] += shared_sum[tid + 2u];
            shared_sum_sq[tid] += shared_sum_sq[tid + 2u];
        }
        workgroupBarrier();
        
        if (tid == 0u) {
            shared_sum[0] += shared_sum[1];
            shared_sum_sq[0] += shared_sum_sq[1];
            
            // Store partial results
            stats[workgroup_id.x * 2u] = shared_sum[0];
            stats[workgroup_id.x * 2u + 1u] = shared_sum_sq[0];
        }
    }
}

// Inline grid dimension helper
fn gridDim() -> vec3<u32> {
    return vec3<u32>(256u, 1u, 1u);  // Max workgroups
}

// Pass 2: Finalize stats + normalize in ONE pass (FUSED!)
@compute @workgroup_size(128)
fn finalize_and_normalize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // First, reduce all partial sums (only first workgroup does this)
    var mean = 0.0;
    var variance = 0.0;
    
    if (gid == 0u) {
        // Thread 0 reduces all partials
        let num_partials = (params.size + 127u) / 128u;  // Num workgroups from pass 1
        var total_sum = 0.0;
        var total_sum_sq = 0.0;
        
        for (var i = 0u; i < num_partials; i = i + 1u) {
            total_sum += stats[i * 2u];
            total_sum_sq += stats[i * 2u + 1u];
        }
        
        let n = f32(params.size);
        mean = total_sum / n;
        variance = (total_sum_sq / n) - (mean * mean);
        
        // Store for other threads
        stats[0] = mean;
        stats[1] = variance;
    }
    
    // Sync to ensure mean/variance are ready
    workgroupBarrier();
    
    // All threads normalize their elements
    mean = stats[0];
    variance = stats[1];
    let inv_std = 1.0 / sqrt(variance + params.epsilon);
    
    // Grid-stride loop for normalization
    var idx = gid;
    while (idx < params.size) {
        let normalized = (input[idx] - mean) * inv_std;
        output[idx] = normalized * gamma[idx] + beta[idx];
        idx += 128u * gridDim().x;
    }
}
