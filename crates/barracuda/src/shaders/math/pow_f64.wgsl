// pow_f64.wgsl — Element-wise power operation (f64 canonical)
//
// General case with full pow_f64 for arbitrary exponents.
// Cross-domain: activation functions, polynomial features, gamma correction.

struct Params {
    size: u32,
    exponent: f64,
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

    // Compute base^exponent
    output[idx] = pow_f64(input[idx], params.exponent);
}
