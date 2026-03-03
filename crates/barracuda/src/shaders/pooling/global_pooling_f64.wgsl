// Global Pooling - Graph-level representation aggregation (f64 canonical)
// Aggregate node features to graph-level representation
// Supports: sum, mean, max aggregation

struct Params {
    num_nodes: u32,
    num_features: u32,
    aggregation_type: u32,  // 0 = sum, 1 = mean, 2 = max
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> node_features: array<f64>;  // [num_nodes, num_features]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;   // [num_features]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let feature_idx = global_id.x;
    if (feature_idx >= params.num_features) {
        return;
    }

    if (params.aggregation_type == 0u) {
        var sum: f64 = 0.0;
        for (var n = 0u; n < params.num_nodes; n = n + 1u) {
            sum = sum + node_features[n * params.num_features + feature_idx];
        }
        output[feature_idx] = sum;

    } else if (params.aggregation_type == 1u) {
        var sum: f64 = 0.0;
        for (var n = 0u; n < params.num_nodes; n = n + 1u) {
            sum = sum + node_features[n * params.num_features + feature_idx];
        }
        output[feature_idx] = sum / f64(params.num_nodes);

    } else if (params.aggregation_type == 2u) {
        var max_val: f64 = -1e308;
        for (var n = 0u; n < params.num_nodes; n = n + 1u) {
            let val = node_features[n * params.num_features + feature_idx];
            max_val = max(max_val, val);
        }
        output[feature_idx] = max_val;
    }
}
