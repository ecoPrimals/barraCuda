// chamfer_distance_f64.wgsl - Chamfer Distance for point clouds (f64 canonical)
//
// Measures similarity between two point clouds
// Used in 3D reconstruction, point cloud generation
//
// CD(X, Y) = (1/|X|) Σ_{x∈X} min_{y∈Y} ||x-y||² + (1/|Y|) Σ_{y∈Y} min_{x∈X} ||y-x||²

struct Params {
    num_points_x: u32,
    num_points_y: u32,
    point_dim: u32,  // Typically 3 for 3D points
    direction: u32,  // 0 = X→Y, 1 = Y→X, 2 = bidirectional
}

@group(0) @binding(0) var<storage, read> points_x: array<f64>;     // [num_points_x, point_dim]
@group(0) @binding(1) var<storage, read> points_y: array<f64>;     // [num_points_y, point_dim]
@group(0) @binding(2) var<storage, read_write> distances: array<f64>; // [num_points_x] or [num_points_y]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (params.direction == 0u || params.direction == 2u) {
        // X → Y: For each point in X, find nearest in Y
        if (idx >= params.num_points_x) {
            return;
        }

        var min_dist: f64 = 1e10;

        for (var j: u32 = 0u; j < params.num_points_y; j = j + 1u) {
            var dist_sq: f64 = 0.0;

            for (var d: u32 = 0u; d < params.point_dim; d = d + 1u) {
                let x_val = points_x[idx * params.point_dim + d];
                let y_val = points_y[j * params.point_dim + d];
                let diff = x_val - y_val;
                dist_sq = dist_sq + diff * diff;
            }

            min_dist = min(min_dist, dist_sq);
        }

        distances[idx] = min_dist;

    } else if (params.direction == 1u) {
        // Y → X: For each point in Y, find nearest in X
        if (idx >= params.num_points_y) {
            return;
        }

        var min_dist: f64 = 1e10;

        for (var i: u32 = 0u; i < params.num_points_x; i = i + 1u) {
            var dist_sq: f64 = 0.0;

            for (var d: u32 = 0u; d < params.point_dim; d = d + 1u) {
                let y_val = points_y[idx * params.point_dim + d];
                let x_val = points_x[i * params.point_dim + d];
                let diff = y_val - x_val;
                dist_sq = dist_sq + diff * diff;
            }

            min_dist = min(min_dist, dist_sq);
        }

        distances[idx] = min_dist;
    }
}
