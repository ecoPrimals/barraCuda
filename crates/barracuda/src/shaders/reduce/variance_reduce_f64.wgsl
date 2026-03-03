// Variance/Std Reduction (f64) - Compute variance and std deviation in double precision
//
// Algorithm: Welford's online algorithm (numerically stable)
// Computes variance in a single pass using mean and M2 accumulators
// Var(X) = M2 / (n - 1) for sample variance, M2 / n for population variance
//
// Use cases:
//   - Statistical analysis
//   - Error estimation
//   - Convergence metrics
//   - Scientific measurements
//
// Deep Debt Principles:
// - Pure WGSL (universal compute, hardware-agnostic)
// - Full f64 precision via SPIR-V/Vulkan
// - Numerically stable (Welford's algorithm)
// - Zero unsafe code
// - Self-contained (no external dependencies)
//
// Notes:
// - Uses Welford's algorithm for numerical stability
// - Parallel merge of partial results preserves stability
// - Returns M2 (sum of squared deviations) and count
// - Final variance = M2 / (count - 1) computed on CPU

struct ReduceParams {
    size: u32,      // Number of elements to reduce
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

// Welford state: (count, mean, M2)
// Packed as: output[3*i] = count, output[3*i+1] = mean, output[3*i+2] = M2
struct WelfordState {
    count: f64,
    mean: f64,
    m2: f64,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // Partial results: [count, mean, M2] per workgroup
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> shared_count: array<f64, 256>;
var<workgroup> shared_mean: array<f64, 256>;
var<workgroup> shared_m2: array<f64, 256>;

// Merge two Welford states (parallel-friendly)
fn merge_welford(
    count_a: f64, mean_a: f64, m2_a: f64,
    count_b: f64, mean_b: f64, m2_b: f64,
) -> WelfordState {
    let count = count_a + count_b;
    if (count == f64(0.0)) {
        return WelfordState(f64(0.0), f64(0.0), f64(0.0));
    }
    let delta = mean_b - mean_a;
    let mean = mean_a + delta * count_b / count;
    let m2 = m2_a + m2_b + delta * delta * count_a * count_b / count;
    return WelfordState(count, mean, m2);
}

// Variance reduction: computes (count, mean, M2) for this workgroup's range
// Final variance = M2 / (count - 1) for sample, M2 / count for population
// Dispatch: (ceil(size / 256), 1, 1)
@compute @workgroup_size(256)
fn variance_reduce_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Initialize Welford state for each thread's element
    if (gid < params.size) {
        let x = input[gid];
        shared_count[tid] = f64(1.0);
        shared_mean[tid] = x;
        shared_m2[tid] = f64(0.0);
    } else {
        // Empty state
        shared_count[tid] = f64(0.0);
        shared_mean[tid] = f64(0.0);
        shared_m2[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction using Welford merge
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let merged = merge_welford(
                shared_count[tid], shared_mean[tid], shared_m2[tid],
                shared_count[tid + stride], shared_mean[tid + stride], shared_m2[tid + stride]
            );
            shared_count[tid] = merged.count;
            shared_mean[tid] = merged.mean;
            shared_m2[tid] = merged.m2;
        }
        workgroupBarrier();
    }

    // Write partial result: [count, mean, M2]
    if (tid == 0u) {
        let base = workgroup_id.x * 3u;
        output[base] = shared_count[0];
        output[base + 1u] = shared_mean[0];
        output[base + 2u] = shared_m2[0];
    }
}

// Sum of squared deviations from a known mean
// Useful for two-pass variance: pass 1 computes mean, pass 2 computes SSD
struct SsdParams {
    size: u32,
    _pad1: u32,
    mean: f64,  // Pre-computed mean (uses 2 f32 slots due to alignment)
}

@group(1) @binding(0) var<storage, read> ssd_input: array<f64>;
@group(1) @binding(1) var<storage, read_write> ssd_output: array<f64>;
@group(1) @binding(2) var<uniform> ssd_params: SsdParams;

var<workgroup> shared_ssd: array<f64, 256>;

@compute @workgroup_size(256)
fn sum_squared_deviation_f64(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    // Compute squared deviation from mean
    if (gid < ssd_params.size) {
        let diff = ssd_input[gid] - ssd_params.mean;
        shared_ssd[tid] = diff * diff;
    } else {
        shared_ssd[tid] = f64(0.0);
    }
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_ssd[tid] = shared_ssd[tid] + shared_ssd[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        ssd_output[workgroup_id.x] = shared_ssd[0];
    }
}
