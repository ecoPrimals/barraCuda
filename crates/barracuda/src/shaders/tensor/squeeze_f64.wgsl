// Squeeze - remove dimensions of size 1 (f64 canonical)
// This is a metadata-only operation in most frameworks
// For GPU, we just copy the data (shape handled by tensor metadata)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= arrayLength(&input)) {
        return;
    }

    output[idx] = input[idx];
}
