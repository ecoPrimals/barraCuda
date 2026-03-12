// SPDX-License-Identifier: AGPL-3.0-only
// message_passing_f64.wgsl - Message Passing for Graph Neural Networks (f64 canonical)
//
// Generic message passing framework
// h_i' = UPDATE(h_i, AGGREGATE({MESSAGE(h_i, h_j, e_ij) : j ∈ N(i)}))

struct Params {
    num_nodes: u32,
    num_edges: u32,
    node_feat_dim: u32,
    edge_feat_dim: u32,
    message_dim: u32,
    aggr_type: u32,  // 0 = sum, 1 = mean, 2 = max
}

@group(0) @binding(0) var<storage, read> node_features: array<f64>;    // [num_nodes, node_feat_dim]
@group(0) @binding(1) var<storage, read> edge_index: array<u32>;       // [num_edges, 2] - (source, target)
@group(0) @binding(2) var<storage, read> edge_features: array<f64>;    // [num_edges, edge_feat_dim]
@group(0) @binding(3) var<storage, read> message_mlp: array<f64>;      // Message network weights
@group(0) @binding(4) var<storage, read> update_mlp: array<f64>;       // Update network weights
@group(0) @binding(5) var<storage, read_write> output: array<f64>;     // [num_nodes, node_feat_dim]
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_node = global_id.x;

    if (target_node >= params.num_nodes) {
        return;
    }

    // Initialize aggregation
    var aggregated: array<f64, 256>; // Max 256 features
    for (var f: u32 = 0u; f < params.node_feat_dim && f < 256u; f = f + 1u) {
        aggregated[f] = select(0.0, -1e10, params.aggr_type == 2u); // 0 for sum/mean, -inf for max
    }

    var neighbor_count: u32 = 0u;

    // Collect messages from neighbors
    for (var e: u32 = 0u; e < params.num_edges; e = e + 1u) {
        let source = edge_index[e * 2u];
        let target_data = edge_index[e * 2u + 1u];

        if (target_data == target_node) {
            neighbor_count = neighbor_count + 1u;

            // Simplified message: just copy source features (full version would use MLP)
            for (var f: u32 = 0u; f < params.node_feat_dim && f < 256u; f = f + 1u) {
                let message = node_features[source * params.node_feat_dim + f];

                if (params.aggr_type == 0u || params.aggr_type == 1u) {
                    // Sum/Mean
                    aggregated[f] = aggregated[f] + message;
                } else {
                    // Max
                    aggregated[f] = max(aggregated[f], message);
                }
            }
        }
    }

    // Finalize aggregation (mean requires division)
    if (params.aggr_type == 1u && neighbor_count > 0u) {
        for (var f: u32 = 0u; f < params.node_feat_dim && f < 256u; f = f + 1u) {
            aggregated[f] = aggregated[f] / f64(neighbor_count);
        }
    }

    // Update node features (simplified - just add aggregated messages)
    for (var f: u32 = 0u; f < params.node_feat_dim && f < 256u; f = f + 1u) {
        output[target_node * params.node_feat_dim + f] =
            node_features[target_node * params.node_feat_dim + f] + aggregated[f];
    }
}
