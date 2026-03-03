// Norm Dim: Compute p-norm along a specific dimension (f64 canonical)
// Similar to PyTorch's torch.norm(dim=N, p=p)
// Formula: (sum(|x|^p))^(1/p) along dimension
// Returns norm values along the specified dimension

struct Params {
    dim_size: u32,
    outer_size: u32,
    inner_size: u32,
    p: f64,  // p-norm parameter
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.outer_size * params.inner_size) {
        return;
    }

    let outer = idx / params.inner_size;
    let inner = idx % params.inner_size;

    var sum_power: f64 = 0.0;

    for (var i = 0u; i < params.dim_size; i = i + 1u) {
        let input_idx = outer * params.dim_size * params.inner_size + i * params.inner_size + inner;
        let value = input[input_idx];
        sum_power = sum_power + pow_f64(abs(value), params.p);
    }

    output[idx] = pow_f64(sum_power, 1.0 / params.p);
}
