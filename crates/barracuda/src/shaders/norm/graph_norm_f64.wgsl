// SPDX-License-Identifier: AGPL-3.0-or-later
// graph_norm_f64.wgsl - Graph Normalization (f64 canonical)
//
// Normalizes node features across the graph
// Similar to batch/layer norm but adapted for graph structure

struct Params {
    num_nodes: u32,
    num_features: u32,
    epsilon: f64,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;           // [num_nodes, num_features]
@group(0) @binding(1) var<storage, read> gamma: array<f64>;           // [num_features] - scale
@group(0) @binding(2) var<storage, read> beta: array<f64>;            // [num_features] - shift
@group(0) @binding(3) var<storage, read_write> output: array<f64>;    // [num_nodes, num_features]
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> shared_mean: array<f64, 256>;
var<workgroup> shared_var: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let f = global_id.x;
    let local_idx = local_id.x;
    
    if (f >= params.num_features) {
        return;
    }
    
    // Compute mean across nodes for this feature
    var local_sum: f64 = 0.0;
    for (var n: u32 = 0u; n < params.num_nodes; n = n + 1u) {
        local_sum = local_sum + input[n * params.num_features + f];
    }
    
    shared_mean[local_idx] = local_sum;
    workgroupBarrier();
    
    // Reduce to get total mean
    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_mean[local_idx] = shared_mean[local_idx] + shared_mean[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let mean = shared_mean[0] / f64(params.num_nodes);
    
    // Compute variance
    var local_var: f64 = 0.0;
    for (var n: u32 = 0u; n < params.num_nodes; n = n + 1u) {
        let diff = input[n * params.num_features + f] - mean;
        local_var = local_var + diff * diff;
    }
    
    shared_var[local_idx] = local_var;
    workgroupBarrier();
    
    // Reduce variance
    stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_var[local_idx] = shared_var[local_idx] + shared_var[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let variance = shared_var[0] / f64(params.num_nodes);
    let std_dev = sqrt_f64(variance + params.epsilon);
    
    // Normalize all nodes for this feature
    for (var n: u32 = 0u; n < params.num_nodes; n = n + 1u) {
        let idx = n * params.num_features + f;
        let normalized = (input[idx] - mean) / std_dev;
        output[idx] = gamma[f] * normalized + beta[f];
    }
}
