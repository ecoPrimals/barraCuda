// LogSumExp - Numerically stable log(sum(exp(x)))
// Computes: log(Σ exp(x_i)) = max(x) + log(Σ exp(x_i - max(x)))
//
// Two-pass algorithm:
// Pass 1: Find maximum value for numerical stability
// Pass 2: Compute sum of exp(x - max) and final result
//
// Used in: Softmax, log-likelihood computations, numerical stability

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

// Note: This is a simplified version that computes logsumexp for the entire array
// A more advanced version would support reduction along specific dimensions

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= 1u) {
        return;
    }
    
    // Find maximum value for numerical stability
    var max_val = -3.40282347e+38; // -FLT_MAX
    for (var i = 0u; i < metadata.size; i = i + 1u) {
        max_val = max(max_val, input[i]);
    }
    
    // Compute sum of exp(x - max)
    var sum = 0.0;
    for (var i = 0u; i < metadata.size; i = i + 1u) {
        sum += exp(input[i] - max_val);
    }
    
    // Final result: max + log(sum)
    output[0] = max_val + log(sum);
}
