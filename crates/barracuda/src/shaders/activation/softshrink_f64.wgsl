// Softshrink - Soft shrinkage activation function (f64 canonical)
// Softshrink: f(x) = x - lambda if x > lambda, x + lambda if x < -lambda, 0 otherwise

struct Params {
    size: u32,
    lambda: f64,
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

    if (x > params.lambda) {
        output[idx] = x - params.lambda;
    } else if (x < -params.lambda) {
        output[idx] = x + params.lambda;
    } else {
        output[idx] = 0.0;
    }
}
