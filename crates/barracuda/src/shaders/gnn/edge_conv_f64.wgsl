// SPDX-License-Identifier: AGPL-3.0-only
// edge_conv.wgsl - Edge Convolution for Graph Neural Networks (f64 canonical)

struct Params {
    num_nodes: u32,
    feature_dim: u32,
    output_dim: u32,
    num_edges: u32,
}

@group(0) @binding(0) var<storage, read> node_features: array<f64>;
@group(0) @binding(1) var<storage, read> edge_offsets: array<f32>;
@group(0) @binding(2) var<storage, read> edge_targets: array<f32>;
@group(0) @binding(3) var<storage, read> mlp_weight: array<f64>;
@group(0) @binding(4) var<storage, read> mlp_bias: array<f64>;
@group(0) @binding(5) var<storage, read_write> output: array<f64>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node_idx = global_id.x;

    if (node_idx >= params.num_nodes) {
        return;
    }

    var center_features: array<f64, 256>;
    for (var f: u32 = 0u; f < params.feature_dim && f < 256u; f = f + 1u) {
        center_features[f] = node_features[node_idx * params.feature_dim + f];
    }

    var max_features: array<f64, 256>;
    for (var o: u32 = 0u; o < params.output_dim && o < 256u; o = o + 1u) {
        max_features[o] = -1e10;
    }

    let neighbor_start = u32(edge_offsets[node_idx]);
    let neighbor_end = u32(edge_offsets[node_idx + 1u]);

    for (var k: u32 = neighbor_start; k < neighbor_end; k = k + 1u) {
        let neighbor_idx = u32(edge_targets[k]);

        if (neighbor_idx >= params.num_nodes) {
            continue;
        }

        var edge_feature: array<f64, 512>;

        for (var f: u32 = 0u; f < params.feature_dim && f < 256u; f = f + 1u) {
            edge_feature[f] = center_features[f];
        }

        for (var f: u32 = 0u; f < params.feature_dim && f < 256u; f = f + 1u) {
            let neighbor_feat = node_features[neighbor_idx * params.feature_dim + f];
            edge_feature[params.feature_dim + f] = neighbor_feat - center_features[f];
        }

        for (var o: u32 = 0u; o < params.output_dim && o < 256u; o = o + 1u) {
            var sum: f64 = 0.0;

            let double_feat = 2u * params.feature_dim;
            for (var f: u32 = 0u; f < double_feat && f < 512u; f = f + 1u) {
                let w_idx = o * double_feat + f;
                sum = sum + mlp_weight[w_idx] * edge_feature[f];
            }

            sum = max(0.0, sum + mlp_bias[o]);

            max_features[o] = max(max_features[o], sum);
        }
    }

    let has_neighbors = neighbor_end > neighbor_start;

    for (var o: u32 = 0u; o < params.output_dim && o < 256u; o = o + 1u) {
        let val = select(0.0, max_features[o], has_neighbors);
        output[node_idx * params.output_dim + o] = val;
    }
}
