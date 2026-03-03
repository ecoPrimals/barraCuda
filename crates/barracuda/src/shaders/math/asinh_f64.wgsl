// Inverse hyperbolic sine operation (f64 canonical)
// asinh(x) = ln(x + sqrt(x² + 1))
// Accepts all real numbers

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }

    let x = input[idx];

    // asinh(x) = ln(x + sqrt(x² + 1))
    output[idx] = log_f64(x + sqrt_f64(x * x + 1.0));
}
