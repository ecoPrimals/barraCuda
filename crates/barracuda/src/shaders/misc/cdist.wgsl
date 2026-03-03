// Cdist - Pairwise distance computation
// Computes distances between all pairs of vectors

struct Params {
    m: u32,        // Number of vectors in A
    n: u32,        // Number of vectors in B
    d: u32,        // Dimensionality of vectors
    metric: u32,   // 0=euclidean, 1=manhattan, 2=cosine
}

@group(0) @binding(0) var<storage, read> input_a: array<f32>;
@group(0) @binding(1) var<storage, read> input_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;  // Index in A
    let j = global_id.y;  // Index in B
    
    if (i >= params.m || j >= params.n) {
        return;
    }
    
    var dist: f32 = 0.0;
    
    // Euclidean distance (L2)
    if (params.metric == 0u) {
        var sum_sq: f32 = 0.0;
        for (var k = 0u; k < params.d; k = k + 1u) {
            let a_val = input_a[i * params.d + k];
            let b_val = input_b[j * params.d + k];
            let diff = a_val - b_val;
            sum_sq = sum_sq + diff * diff;
        }
        dist = sqrt(sum_sq);
    }
    // Manhattan distance (L1)
    else if (params.metric == 1u) {
        var sum_abs: f32 = 0.0;
        for (var k = 0u; k < params.d; k = k + 1u) {
            let a_val = input_a[i * params.d + k];
            let b_val = input_b[j * params.d + k];
            sum_abs = sum_abs + abs(a_val - b_val);
        }
        dist = sum_abs;
    }
    // Cosine distance (1 - cosine_similarity)
    else if (params.metric == 2u) {
        var dot: f32 = 0.0;
        var norm_a: f32 = 0.0;
        var norm_b: f32 = 0.0;
        for (var k = 0u; k < params.d; k = k + 1u) {
            let a_val = input_a[i * params.d + k];
            let b_val = input_b[j * params.d + k];
            dot = dot + a_val * b_val;
            norm_a = norm_a + a_val * a_val;
            norm_b = norm_b + b_val * b_val;
        }
        let cosine_sim = dot / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
        dist = 1.0 - cosine_sim;
    }
    
    output[i * params.n + j] = dist;
}
