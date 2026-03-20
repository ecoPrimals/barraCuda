// SPDX-License-Identifier: AGPL-3.0-or-later
// GELU - Gaussian Error Linear Unit (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> size: u32;

// Constants for GELU approximation
const SQRT_2_OVER_PI: f64 = 0.7978845608;  // sqrt(2/pi)
const GELU_COEFF: f64 = 0.044715;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let x = input[idx];
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let x_cubed = x * x * x;
    let inner = SQRT_2_OVER_PI * (x + GELU_COEFF * x_cubed);
    output[idx] = f64(0.5) * x * (f64(1.0) + tanh_f64(inner));
}
