// SPDX-License-Identifier: AGPL-3.0-only
// graph_conv.wgsl - Graph Convolution (f64 canonical)

struct Params {
    num_nodes: u32,
    in_features: u32,
    out_features: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> node_features: array<f64>;
@group(0) @binding(1) var<storage, read> adj_matrix: array<f64>;
@group(0) @binding(2) var<storage, read> weight: array<f64>;
@group(0) @binding(3) var<storage, read> bias: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;
    
    if (node_idx >= params.num_nodes) {
        return;
    }
    
    for (var out_f: u32 = 0u; out_f < params.out_features; out_f = out_f + 1u) {
        var sum: f64 = 0.0;
        
        for (var neighbor: u32 = 0u; neighbor < params.num_nodes; neighbor = neighbor + 1u) {
            let adj_weight = adj_matrix[node_idx * params.num_nodes + neighbor];
            
            if (adj_weight > 0.0) {
                for (var in_f: u32 = 0u; in_f < params.in_features; in_f = in_f + 1u) {
                    let feat = node_features[neighbor * params.in_features + in_f];
                    let w = weight[in_f * params.out_features + out_f];
                    sum = sum + adj_weight * feat * w;
                }
            }
        }
        
        sum = sum + bias[out_f];
        sum = max(0.0, sum);
        
        output[node_idx * params.out_features + out_f] = sum;
    }
}
