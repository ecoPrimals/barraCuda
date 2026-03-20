// SPDX-License-Identifier: AGPL-3.0-or-later
// Optimized element-wise addition: C = A + B
// 
// Optimizations:
// 1. Process 4 elements per thread (vec4 vectorization)
// 2. Workgroup size 64 (better for AMD wavefronts, fine for NVIDIA)
// 3. Uniform buffer for size (avoid arrayLength() overhead)
// 4. Unrolled inner loop for ILP

struct Params {
    size: u32,        // Total elements (must be multiple of 4)
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
    
    // Process 4 elements as vec4 (coalesced 16-byte load/store)
    if (idx < vec_count) {
        output[idx] = a[idx] + b[idx];
    }
}

// Variant: Process 8 elements per thread for higher ILP
@compute @workgroup_size(64)
fn main_8x(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let base_idx = global_id.x * 2u;
    let vec_count = params.size / 4u;
    
    // First vec4
    if (base_idx < vec_count) {
        output[base_idx] = a[base_idx] + b[base_idx];
    }
    
    // Second vec4 (8 elements total per thread)
    let idx2 = base_idx + 1u;
    if (idx2 < vec_count) {
        output[idx2] = a[idx2] + b[idx2];
    }
}
