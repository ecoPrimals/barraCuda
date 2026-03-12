// SPDX-License-Identifier: AGPL-3.0-only
// SAGEConv - GraphSAGE (f64 canonical)

struct Params {
    num_nodes: u32,
    num_edges: u32,
    in_features: u32,
    out_features: u32,
    normalize: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> node_features: array<f64>;
@group(0) @binding(2) var<storage, read> edge_index: array<u32>;
@group(0) @binding(3) var<storage, read> weights: array<f64>;
@group(0) @binding(4) var<storage, read> degrees: array<u32>;
@group(0) @binding(5) var<storage, read_write> aggregated: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn aggregate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let edge = global_id.x;
    if (edge >= params.num_edges) {
        return;
    }

    let src = edge_index[edge * 2u];
    let dst = edge_index[edge * 2u + 1u];

    for (var f = 0u; f < params.in_features; f = f + 1u) {
        let val = node_features[src * params.in_features + f];
        atomicAdd(&aggregated[dst * params.in_features + f], bitcast<i32>(f32(val)));
    }
}

@compute @workgroup_size(256)
fn apply_transform(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node = global_id.x;
    if (node >= params.num_nodes) {
        return;
    }

    let deg = f64(degrees[node]);
    
    for (var out_f = 0u; out_f < params.out_features; out_f = out_f + 1u) {
        var sum: f64 = 0.0;
        
        for (var in_f = 0u; in_f < params.in_features; in_f = in_f + 1u) {
            let feat = node_features[node * params.in_features + in_f];
            let weight_idx = in_f * params.out_features + out_f;
            sum += feat * weights[weight_idx];
        }
        
        for (var in_f = 0u; in_f < params.in_features; in_f = in_f + 1u) {
            var aggr = f64(bitcast<f32>(atomicLoad(&aggregated[node * params.in_features + in_f])));
            if (deg > 0.0) {
                aggr /= deg;
            }
            let weight_idx = (params.in_features + in_f) * params.out_features + out_f;
            sum += aggr * weights[weight_idx];
        }
        
        output[node * params.out_features + out_f] = sum;
    }
}

@compute @workgroup_size(256)
fn normalize_output(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let node = global_id.x;
    if (node >= params.num_nodes || params.normalize == 0u) {
        return;
    }

    var norm_sq: f64 = 0.0;
    for (var f = 0u; f < params.out_features; f = f + 1u) {
        let val = output[node * params.out_features + f];
        norm_sq += val * val;
    }
    let norm = sqrt_f64(norm_sq) + 1e-8;

    for (var f = 0u; f < params.out_features; f = f + 1u) {
        let idx = node * params.out_features + f;
        output[idx] /= norm;
    }
}
