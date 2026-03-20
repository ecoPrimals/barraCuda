// SPDX-License-Identifier: AGPL-3.0-or-later
// cosine_similarity.wgsl - Cosine Similarity
//
// Computes cosine similarity between pairs of vectors
// Similarity = (a · b) / (||a|| * ||b||)
//
// Used in recommendation systems, similarity search, clustering

struct Params {
    num_vectors_a: u32,
    num_vectors_b: u32,
    vector_dim: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> vectors_a: array<f32>;     // [num_vectors_a, dim]
@group(0) @binding(1) var<storage, read> vectors_b: array<f32>;     // [num_vectors_b, dim]
@group(0) @binding(2) var<storage, read_write> output: array<f32>;  // [num_vectors_a, num_vectors_b]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    
    if (i >= params.num_vectors_a || j >= params.num_vectors_b) {
        return;
    }
    
    var dot_product: f32 = 0.0;
    var norm_a: f32 = 0.0;
    var norm_b: f32 = 0.0;
    
    for (var d: u32 = 0u; d < params.vector_dim; d = d + 1u) {
        let a_val = vectors_a[i * params.vector_dim + d];
        let b_val = vectors_b[j * params.vector_dim + d];
        
        dot_product = dot_product + a_val * b_val;
        norm_a = norm_a + a_val * a_val;
        norm_b = norm_b + b_val * b_val;
    }
    
    // Cosine similarity
    let similarity = dot_product / (sqrt(norm_a) * sqrt(norm_b) + 1e-8);
    
    output[i * params.num_vectors_b + j] = similarity;
}
