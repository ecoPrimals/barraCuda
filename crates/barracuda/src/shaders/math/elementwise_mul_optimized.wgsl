// SPDX-License-Identifier: AGPL-3.0-only
// Optimized element-wise multiplication: C = A * B
// 
// Optimizations:
// 1. Process 4 elements per thread (vec4 vectorization)
// 2. Workgroup size 64 (AMD wavefront aligned)
// 3. Uniform buffer for size

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let vec_count = params.size / 4u;
    
    if (idx < vec_count) {
        output[idx] = a[idx] * b[idx];
    }
}
