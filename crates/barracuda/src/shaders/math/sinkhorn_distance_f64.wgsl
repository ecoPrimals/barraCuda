// sinkhorn_distance_f64.wgsl - Sinkhorn Distance (Regularized Optimal Transport) (f64 canonical)
//
// Approximates Wasserstein distance using Sinkhorn iterations
// Used for comparing distributions with efficient computation

struct Params {
    size: u32,
    num_iterations: u32,
    epsilon: f64,      // Regularization parameter
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> dist1: array<f64>;        // Source distribution
@group(0) @binding(1) var<storage, read> dist2: array<f64>;        // Target distribution
@group(0) @binding(2) var<storage, read> cost_matrix: array<f64>;  // [size, size] - pairwise costs
@group(0) @binding(3) var<storage, read_write> transport: array<f64>; // [size, size] - transport plan
@group(0) @binding(4) var<storage, read_write> output: array<f64>;    // [1] - distance
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_u: array<f64, 256>;
var<workgroup> shared_v: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    if (idx >= params.size) {
        return;
    }

    // Initialize dual variables
    if (idx < params.size) {
        shared_u[idx] = 1.0;
        shared_v[idx] = 1.0;
    }

    workgroupBarrier();

    // Sinkhorn iterations
    for (var iter: u32 = 0u; iter < params.num_iterations; iter = iter + 1u) {
        // Update u
        if (idx < params.size) {
            var sum: f64 = 0.0;
            for (var j: u32 = 0u; j < params.size; j = j + 1u) {
                let cost = cost_matrix[idx * params.size + j];
                let kernel = exp_f64(-cost / params.epsilon);
                sum = sum + kernel * shared_v[j];
            }
            shared_u[idx] = dist1[idx] / (sum + 1e-12);
        }

        workgroupBarrier();

        // Update v
        if (idx < params.size) {
            var sum: f64 = 0.0;
            for (var i: u32 = 0u; i < params.size; i = i + 1u) {
                let cost = cost_matrix[i * params.size + idx];
                let kernel = exp_f64(-cost / params.epsilon);
                sum = sum + kernel * shared_u[i];
            }
            shared_v[idx] = dist2[idx] / (sum + 1e-12);
        }

        workgroupBarrier();
    }

    // Compute transport plan and distance
    if (idx == 0u) {
        var total_cost: f64 = 0.0;

        for (var i: u32 = 0u; i < params.size; i = i + 1u) {
            for (var j: u32 = 0u; j < params.size; j = j + 1u) {
                let cost = cost_matrix[i * params.size + j];
                let kernel = exp_f64(-cost / params.epsilon);
                let t = shared_u[i] * kernel * shared_v[j];

                transport[i * params.size + j] = t;
                total_cost = total_cost + t * cost;
            }
        }

        output[0] = total_cost;
    }
}
