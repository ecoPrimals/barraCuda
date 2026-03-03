// Softsign - x / (1 + |x|) (f64 canonical)

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
    output[idx] = x / (1.0 + abs(x));
}
