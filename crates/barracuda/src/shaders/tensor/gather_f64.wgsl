// Gather - Select elements from input using indices (f64 canonical)
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
    gather_size: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Decompose output index
    let outer = idx / (params.gather_size * params.inner_size);
    let mid = (idx / params.inner_size) % params.gather_size;
    let inner = idx % params.inner_size;

    // Get the index to gather from
    let gather_idx = indices[mid];

    // Bounds check
    if (gather_idx >= params.dim_size) {
        output[idx] = 0.0;
        return;
    }

    // Calculate input position
    let input_idx = outer * params.dim_size * params.inner_size +
                    gather_idx * params.inner_size + inner;

    output[idx] = input[input_idx];
}
