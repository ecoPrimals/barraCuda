// SPDX-License-Identifier: AGPL-3.0-or-later
// Slice Assign - In-place slice assignment with strided writes (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    input_size: u32,
    start: u32,
    end: u32,
    stride: u32,
    values_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> values: array<f64>;
@group(0) @binding(2) var<storage, read_write> input: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.values_size) {
        return;
    }
    
    // Calculate target position in input: start + idx * stride
    let target_data_idx = params.start + idx * params.stride;
    
    // Bounds check
    if (target_data_idx < params.end && target_data_idx < params.input_size) {
        input[target_data_idx] = values[idx];
    }
}
