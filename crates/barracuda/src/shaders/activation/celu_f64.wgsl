// CELU - Continuously Differentiable Exponential Linear Unit (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    size: u32,
    alpha: f64,
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

    let x = input[idx];
    if (x >= f64(0.0)) {
        output[idx] = x;
    } else {
        output[idx] = params.alpha * (exp_f64(x / params.alpha) - f64(1.0));
    }
}
