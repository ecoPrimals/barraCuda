// LayerNorm: Layer normalization with Welford's algorithm (f64 canonical)
// CUDA equivalent: cudnn::LayerNormalization
// Formula: output = (input - mean) / sqrt(variance + epsilon) * gamma + beta
// Use cases: Transformer normalization, stabilizing training

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> gamma: array<f64>;  // Scale (optional, can be all 1s)
@group(0) @binding(2) var<storage, read> beta: array<f64>;   // Shift (optional, can be all 0s)
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<storage, read_write> stats: array<f64>;  // [mean, variance]

struct Params {
    size: u32,
    epsilon: f64,
}
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> shared_sum: array<f64, 256>;
var<workgroup> shared_sum_sq: array<f64, 256>;

// Pass 1: Compute mean and variance using Welford's online algorithm
@compute @workgroup_size(256)
fn compute_stats(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Load and compute local statistics
    var value: f64 = 0.0;
    var value_sq: f64 = 0.0;
    if (gid < params.size) {
        value = input[gid];
        value_sq = value * value;
    }
    shared_sum[tid] = value;
    shared_sum_sq[tid] = value_sq;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        stats[workgroup_id.x * 2u] = shared_sum[0];  // Partial sum
        stats[workgroup_id.x * 2u + 1u] = shared_sum_sq[0];  // Partial sum of squares
    }
}

// Pass 2: Finalize statistics (reduce partial sums to final mean/variance)
@compute @workgroup_size(256)
fn finalize_stats(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;

    // Load partial sums from multiple workgroups
    var local_sum: f64 = 0.0;
    var local_sum_sq: f64 = 0.0;

    // Each thread loads one partial sum
    let num_partials = arrayLength(&stats) / 2u;
    if (tid < num_partials) {
        local_sum = stats[tid * 2u];
        local_sum_sq = stats[tid * 2u + 1u];
    }
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    workgroupBarrier();

    // Tree reduction to sum all partials
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (tid + stride) < num_partials) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
            shared_sum_sq[tid] = shared_sum_sq[tid] + shared_sum_sq[tid + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 computes final mean and variance
    if (tid == 0u) {
        let total_sum = shared_sum[0];
        let total_sum_sq = shared_sum_sq[0];
        let n = f64(params.size);

        let mean = total_sum / n;
        let variance = (total_sum_sq / n) - (mean * mean);

        // Store final stats at beginning of buffer
        stats[0] = mean;
        stats[1] = variance;
    }
}

// Pass 3: Normalize
@compute @workgroup_size(256)
fn normalize(
    @builtin(global_invocation_id) global_id: vec3<u32>,
) {
    let gid = global_id.x;

    if (gid < params.size) {
        // Final mean and variance are now in stats[0] and stats[1]
        let mean = stats[0];
        let variance = stats[1];

        let normalized = (input[gid] - mean) / sqrt_f64(variance + params.epsilon);
        output[gid] = normalized * gamma[gid] + beta[gid];
    }
}
