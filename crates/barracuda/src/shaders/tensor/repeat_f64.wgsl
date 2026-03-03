// Repeat - Repeat tensor along dimensions (f64 canonical)
//
// Deep Debt Principles:
// - Pure WGSL implementation (universal compute)
// - Zero unsafe code (memory safe)
// - Hardware-agnostic (works on any GPU/CPU via WebGPU)
// - Self-contained logic (no external dependencies)

struct Params {
    input_size: u32,
    output_size: u32,
    num_dims: u32,
    _pad: u32,
    dim_sizes: vec4<u32>,
    repeats: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.output_size) {
        return;
    }

    // Calculate input index by mapping output index back through repeats
    var input_idx = 0u;
    var remaining = idx;
    var stride = 1u;

    for (var i = 0u; i < params.num_dims; i = i + 1u) {
        let dim_idx = params.num_dims - 1u - i;
        let output_dim_size = params.dim_sizes[dim_idx] * params.repeats[dim_idx];
        let pos = remaining % output_dim_size;
        let input_pos = pos / params.repeats[dim_idx];

        input_idx += input_pos * stride;
        stride *= params.dim_sizes[dim_idx];
        remaining /= output_dim_size;
    }

    output[idx] = input[input_idx];
}
