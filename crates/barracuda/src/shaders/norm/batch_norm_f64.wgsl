// BatchNorm - Batch normalization (f64 canonical)
// output = (input - mean) / sqrt(variance + epsilon) * gamma + beta
// Simplified version: per-tensor normalization with default gamma=1, beta=0

struct BatchNormParams {
    epsilon: f64,
    _padding: vec3<f64>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: BatchNormParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let size = arrayLength(&input);

    if (idx >= size) {
        return;
    }

    // Compute mean (simplified: single pass for small tensors)
    if (idx == 0u) {
        var sum = f64(0.0);
        for (var i = 0u; i < size; i = i + 1u) {
            sum = sum + input[i];
        }
        let mean = sum / f64(size);

        // Compute variance
        var variance = f64(0.0);
        for (var i = 0u; i < size; i = i + 1u) {
            let diff = input[i] - mean;
            variance = variance + diff * diff;
        }
        variance = variance / f64(size);

        // Normalize all elements (gamma=1, beta=0 for simplicity)
        let std_dev = sqrt_f64(variance + params.epsilon);
        for (var i = 0u; i < size; i = i + 1u) {
            output[i] = (input[i] - mean) / std_dev;
        }
    }
}
