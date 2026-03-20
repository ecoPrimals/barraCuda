// SPDX-License-Identifier: AGPL-3.0-or-later
// Scatter - Write values to specific indices (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    size: u32,
    dim_size: u32,
    outer_size: u32,
    inner_size: u32,
    scatter_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> values: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // First, copy input to output
    output[idx] = input[idx];
}

// Second pass: scatter values
@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.outer_size * params.scatter_size * params.inner_size) {
        return;
    }

    // Decompose scatter index
    let outer = idx / (params.scatter_size * params.inner_size);
    let mid = (idx / params.inner_size) % params.scatter_size;
    let inner = idx % params.inner_size;

    // Get the index to scatter to
    let scatter_idx = indices[mid];

    // Bounds check
    if (scatter_idx >= params.dim_size) {
        return;
    }

    // Calculate output position
    let output_idx = outer * params.dim_size * params.inner_size +
                     scatter_idx * params.inner_size + inner;

    // Write the value (atomic not needed in WebGPU for non-overlapping writes)
    output[output_idx] = values[idx];
}
