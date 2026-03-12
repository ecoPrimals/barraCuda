// SPDX-License-Identifier: AGPL-3.0-only
// Cumsum F64 - Cumulative sum along a dimension (double precision)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU with SHADER_F64)
// - Self-contained logic (no external dependencies)
//
// Algorithm: Per-thread sequential scan along dimension
// Each thread handles one (outer, inner) coordinate pair
// and scans sequentially along the specified dimension.
//
// For large dim_size, consider hierarchical Blelloch scan.

struct Params {
    size: u32,       // Total number of elements
    dim_size: u32,   // Size of the dimension to scan along
    outer_size: u32, // Product of dimensions before dim
    inner_size: u32, // Product of dimensions after dim
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    // Each thread handles one (outer, inner) coordinate pair
    let total_pairs = params.outer_size * params.inner_size;
    if (idx >= total_pairs) {
        return;
    }
    
    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;
    
    // Inclusive scan along dimension
    // Initialize sum to f64 zero via input subtraction (Naga f64 constant pattern)
    var sum = input[0] - input[0];  // Produces f64 zero
    for (var i = 0u; i < params.dim_size; i++) {
        let linear_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        sum = sum + input[linear_idx];
        output[linear_idx] = sum;
    }
}
