// Narrow - Select a slice along a dimension (f64 canonical)
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
    start: u32,
    length: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.outer_size * params.length * params.inner_size) {
        return;
    }

    let outer = idx / (params.length * params.inner_size);
    let mid = (idx / params.inner_size) % params.length;
    let inner = idx % params.inner_size;

    // Map to input index with offset
    let input_idx = outer * params.dim_size * params.inner_size +
                    (params.start + mid) * params.inner_size + inner;

    output[idx] = input[input_idx];
}
