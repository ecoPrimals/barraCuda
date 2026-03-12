// SPDX-License-Identifier: AGPL-3.0-only
// One-hot - Convert indices to one-hot vectors (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    num_indices: u32,
    num_classes: u32,
}

@group(0) @binding(0) var<storage, read> indices: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.num_indices * params.num_classes) {
        return;
    }
    
    let batch_idx = idx / params.num_classes;
    let class_idx = idx % params.num_classes;
    
    let target_class = indices[batch_idx];
    
    if (class_idx == target_class) {
        output[idx] = 1.0;
    } else {
        output[idx] = 0.0;
    }
}
