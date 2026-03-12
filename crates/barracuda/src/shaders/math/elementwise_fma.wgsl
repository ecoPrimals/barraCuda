// SPDX-License-Identifier: AGPL-3.0-only
// Fused multiply-add: C = A * B + C (or C = alpha * A + B)
// 
// FMA is a single instruction on modern GPUs - 2 FLOPs in 1 op
// This avoids the overhead of separate mul + add dispatches

struct Params {
    size: u32,
    alpha: f32,       // Scalar multiplier (for axpy: C = alpha*A + B)
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

// Standard FMA: C = A * B + C_prev (reads from output, writes back)
@compute @workgroup_size(64)
fn fma_inplace(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_count = params.size / 4u;
    
    if (idx < vec_count) {
        // fma(a, b, c) = a * b + c
        output[idx] = fma(a[idx], b[idx], output[idx]);
    }
}

// AXPY: C = alpha * A + B (BLAS-style)
@compute @workgroup_size(64)
fn axpy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_count = params.size / 4u;
    
    if (idx < vec_count) {
        let alpha_vec = vec4<f32>(params.alpha);
        output[idx] = fma(alpha_vec, a[idx], b[idx]);
    }
}

// Hadamard + Add: C = A * B + D (4 input variant)
@compute @workgroup_size(64)
fn mul_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_count = params.size / 4u;
    
    if (idx < vec_count) {
        output[idx] = fma(a[idx], b[idx], output[idx]);
    }
}
