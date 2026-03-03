// Variance and standard deviation computation — f64 precision
// Var(X) = E[(X-μ)²] = Σ(x-μ)² / (n - ddof)
//
// Input: vector of values (f64)
// Output: variance (or std dev based on mode)
//
// Applications: statistics, normalization, feature scaling
// Reference: Standard statistical formulas

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,        // Length of each vector
    num_vectors: u32, // Number of vectors to process
    stride: u32,      // Stride between vectors
    ddof: u32,        // Delta degrees of freedom (0=population, 1=sample)
    mode: u32,        // 0=variance, 1=std_dev
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_vectors) {
        return;
    }

    let size = params.size;
    let ddof = params.ddof;

    if (size <= ddof) {
        output[idx] = f64(0.0);
        return;
    }

    let offset = idx * params.stride;

    // Pass 1: compute mean
    var sum: f64 = f64(0.0);
    for (var i = 0u; i < size; i = i + 1u) {
        sum = sum + input[offset + i];
    }
    let mean = sum / f64(size);

    // Pass 2: compute variance
    var var_sum: f64 = f64(0.0);
    for (var i = 0u; i < size; i = i + 1u) {
        let d = input[offset + i] - mean;
        var_sum = var_sum + d * d;
    }

    var result = var_sum / f64(size - ddof);

    if (params.mode == 1u) {
        result = sqrt(result);  // Standard deviation
    }

    output[idx] = result;
}
