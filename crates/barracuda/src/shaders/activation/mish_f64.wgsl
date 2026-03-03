// Mish - Self-regularizing activation function (f64 canonical)
// mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^x))

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let x = input[idx];

    // softplus(x) = ln(1 + e^x)
    // For numerical stability, use different formula for large x
    var softplus_val: f64;
    if (x > f64(20.0)) {
        softplus_val = x;  // For large x, ln(1 + e^x) ≈ x
    } else {
        softplus_val = log_f64(f64(1.0) + exp_f64(x));
    }

    // mish(x) = x * tanh(softplus(x))
    let result = x * tanh_f64(softplus_val);

    output[idx] = result;
}
