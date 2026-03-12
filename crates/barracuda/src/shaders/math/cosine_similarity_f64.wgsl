// SPDX-License-Identifier: AGPL-3.0-only
// cosine_similarity_f64.wgsl - Cosine Similarity at f64 precision
//
// Computes cosine similarity between pairs of vectors using f64 arithmetic
// Similarity = (a · b) / (||a|| * ||b||)
//
// REQUIRES: SHADER_F64 feature (wgpu::Features::SHADER_F64)
//
// Use cases (wetSpring Priority 2):
// - MS2 spectral matching in analytical chemistry
// - High-precision similarity search
// - Biological sequence comparison
//
// Date: February 16, 2026
// License: AGPL-3.0-only

// Import sqrt_f64 from math_f64.wgsl (include via ShaderTemplate or inline)
// For standalone use, inline the sqrt function:

fn sqrt_f64_inline(x: f64) -> f64 {
    let zero = x - x;
    if (x <= zero) {
        return zero;
    }
    
    var y = x;
    var scale = zero + 1.0;
    let large = zero + 1e32;
    let small = zero + 1e-32;
    
    if (x > large) {
        y = x / large;
        scale = zero + 1e16;
    } else if (x < small) {
        y = x * large;
        scale = zero + 1e-16;
    }
    
    var r = (y + (zero + 1.0)) / (zero + 2.0);
    let half = zero + 0.5;
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    r = half * (r + y / r);
    
    return r * scale;
}

struct Params {
    num_vectors_a: u32,
    num_vectors_b: u32,
    vector_dim: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> vectors_a: array<f64>;     // [num_vectors_a, dim]
@group(0) @binding(1) var<storage, read> vectors_b: array<f64>;     // [num_vectors_b, dim]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // [num_vectors_a, num_vectors_b]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    
    if (i >= params.num_vectors_a || j >= params.num_vectors_b) {
        return;
    }
    
    // Use (zero + literal) pattern for f64 constants
    let first_a = vectors_a[i * params.vector_dim];
    let zero = first_a - first_a;
    
    var dot_product = zero;
    var norm_a = zero;
    var norm_b = zero;
    
    for (var d: u32 = 0u; d < params.vector_dim; d = d + 1u) {
        let a_val = vectors_a[i * params.vector_dim + d];
        let b_val = vectors_b[j * params.vector_dim + d];
        
        dot_product = dot_product + a_val * b_val;
        norm_a = norm_a + a_val * a_val;
        norm_b = norm_b + b_val * b_val;
    }
    
    // Cosine similarity with f64 precision epsilon
    let epsilon = zero + 1e-15;
    let denom = sqrt_f64_inline(norm_a) * sqrt_f64_inline(norm_b) + epsilon;
    let similarity = dot_product / denom;
    
    output[i * params.num_vectors_b + j] = similarity;
}

// ============================================================================
// Variant: Single vector pair (more efficient for small comparisons)
// ============================================================================

struct SinglePairParams {
    vector_dim: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> vector_a_single: array<f64>;      // [dim]
@group(0) @binding(1) var<storage, read> vector_b_single: array<f64>;      // [dim]
@group(0) @binding(2) var<storage, read_write> result_single: array<f64>;  // [1]
@group(0) @binding(3) var<uniform> single_params: SinglePairParams;

// Workgroup shared memory for parallel reduction
var<workgroup> shared_dot: array<f64, 256>;
var<workgroup> shared_norm_a: array<f64, 256>;
var<workgroup> shared_norm_b: array<f64, 256>;

@compute @workgroup_size(256)
fn cosine_single_pair(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>
) {
    let tid = local_id.x;
    let dim = single_params.vector_dim;
    
    // Each thread handles multiple elements
    let first = vector_a_single[0];
    let zero = first - first;
    
    var local_dot = zero;
    var local_norm_a = zero;
    var local_norm_b = zero;
    
    var idx = tid;
    while (idx < dim) {
        let a_val = vector_a_single[idx];
        let b_val = vector_b_single[idx];
        
        local_dot = local_dot + a_val * b_val;
        local_norm_a = local_norm_a + a_val * a_val;
        local_norm_b = local_norm_b + b_val * b_val;
        
        idx = idx + 256u;
    }
    
    // Store to shared memory
    shared_dot[tid] = local_dot;
    shared_norm_a[tid] = local_norm_a;
    shared_norm_b[tid] = local_norm_b;
    workgroupBarrier();
    
    // Parallel reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_dot[tid] = shared_dot[tid] + shared_dot[tid + stride];
            shared_norm_a[tid] = shared_norm_a[tid] + shared_norm_a[tid + stride];
            shared_norm_b[tid] = shared_norm_b[tid] + shared_norm_b[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Thread 0 writes result
    if (tid == 0u) {
        let epsilon = zero + 1e-15;
        let denom = sqrt_f64_inline(shared_norm_a[0]) * sqrt_f64_inline(shared_norm_b[0]) + epsilon;
        result_single[0] = shared_dot[0] / denom;
    }
}
