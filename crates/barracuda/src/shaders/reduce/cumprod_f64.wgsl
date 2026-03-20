// SPDX-License-Identifier: AGPL-3.0-or-later
// Cumulative Product (f64) - Compute cumulative product along a dimension
//
// For input [a, b, c, d], output is [a, a*b, a*b*c, a*b*c*d]
//
// Use cases:
//   - Probability chains
//   - Running products
//   - Gamma function computation
//   - Factorial approximations
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Zero unsafe code
// - Self-contained (no external dependencies)
//
// Notes:
// - Sequential scan along dimension (parallel across slices)
// - Each thread handles one outer*inner slice
// - Numerically sensitive for long sequences (consider log domain)

struct CumprodParams {
    size: u32,
    dim_size: u32,
    outer_size: u32,
    inner_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: CumprodParams;

// Cumulative product along dimension
// Each thread processes one (outer, inner) slice
// Dispatch: (outer_size * inner_size, 1, 1)
@compute @workgroup_size(256)
fn cumprod_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.outer_size * params.inner_size) {
        return;
    }
    
    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;
    
    var product = f64(1.0);
    for (var i = 0u; i < params.dim_size; i++) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        product = product * input[input_idx];
        output[input_idx] = product;
    }
}

// Reverse cumulative product (from end to start)
// For input [a, b, c, d], output is [a*b*c*d, b*c*d, c*d, d]
@compute @workgroup_size(256)
fn cumprod_reverse_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.outer_size * params.inner_size) {
        return;
    }
    
    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;
    
    var product = f64(1.0);
    for (var i = params.dim_size; i > 0u; i--) {
        let j = i - 1u;
        let input_idx = outer * params.dim_size * params.inner_size + j * params.inner_size + inner;
        product = product * input[input_idx];
        output[input_idx] = product;
    }
}

// Exclusive cumulative product (shifted, starts with 1)
// For input [a, b, c, d], output is [1, a, a*b, a*b*c]
@compute @workgroup_size(256)
fn cumprod_exclusive_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.outer_size * params.inner_size) {
        return;
    }
    
    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;
    
    var product = f64(1.0);
    for (var i = 0u; i < params.dim_size; i++) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        let current = input[input_idx];
        output[input_idx] = product;
        product = product * current;
    }
}

// Log-domain cumulative product (numerically stable for long sequences)
// Computes cumsum(log(x)), caller takes exp() for actual cumprod
// Avoids overflow/underflow for very large/small products
@compute @workgroup_size(256)
fn cumprod_log_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= params.outer_size * params.inner_size) {
        return;
    }
    
    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;
    
    var log_sum = f64(0.0);
    for (var i = 0u; i < params.dim_size; i++) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        let val = input[input_idx];
        if (val > f64(0.0)) {
            log_sum = log_sum + log(val);
        } else if (val == f64(0.0)) {
            // log(0) = -inf, subsequent products are 0
            log_sum = f64(-1e308);
        } else {
            // Negative values - use NaN to signal error
            log_sum = f64(0.0) / f64(0.0);
        }
        output[input_idx] = log_sum;
    }
}
