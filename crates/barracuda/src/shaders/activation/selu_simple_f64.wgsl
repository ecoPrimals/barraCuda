// SELU (Scaled Exponential Linear Unit) - Simple version (f64 canonical)
// Formula: SELU(x) = λ * (x if x > 0, else α(e^x - 1))
// Constants: λ ≈ 1.0507, α ≈ 1.67326

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

const LAMBDA: f64 = 1.0507009873554804934193349852946;
const ALPHA: f64 = 1.6732632423543772848170429916717;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= arrayLength(&input) {
        return;
    }

    let x = input[idx];

    if x > f64(0.0) {
        output[idx] = LAMBDA * x;
    } else {
        output[idx] = LAMBDA * ALPHA * (exp_f64(x) - f64(1.0));
    }
}
