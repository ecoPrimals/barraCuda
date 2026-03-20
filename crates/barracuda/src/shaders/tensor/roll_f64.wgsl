// SPDX-License-Identifier: AGPL-3.0-or-later
// Roll - Shift elements along a dimension (f64 canonical)
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
    shift: i32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Decompose index
    let outer = idx / (params.dim_size * params.inner_size);
    let mid = (idx / params.inner_size) % params.dim_size;
    let inner = idx % params.inner_size;

    // Apply shift with wrapping
    var shifted_mid = (i32(mid) + params.shift) % i32(params.dim_size);
    if (shifted_mid < 0) {
        shifted_mid += i32(params.dim_size);
    }

    let output_idx = outer * params.dim_size * params.inner_size + u32(shifted_mid) * params.inner_size + inner;

    output[output_idx] = input[idx];
}
