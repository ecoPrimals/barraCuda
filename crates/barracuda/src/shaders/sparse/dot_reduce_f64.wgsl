// Dot product and reduction operations for sparse solvers - f64 Precision
// Separate shader to avoid binding conflicts with other sparse operations

struct DotParams {
    n: u32,
}

@group(0) @binding(0) var<storage, read> dot_a: array<f64>;
@group(0) @binding(1) var<storage, read> dot_b: array<f64>;
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f64>;
@group(0) @binding(3) var<uniform> dot_params: DotParams;

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn dot_f64(@builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(workgroup_id) wg_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    let n = dot_params.n;

    // Each thread sums its elements
    var sum: f64 = 0.0;
    var i = gid;
    while (i < n) {
        sum = sum + dot_a[i] * dot_b[i];
        i = i + 256u * 256u;  // Stride by total threads
    }

    shared_sum[tid] = sum;
    workgroupBarrier();

    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        partial_sums[wg_id.x] = shared_sum[0];
    }
}

// Final reduction: sum partial_sums[0..n_workgroups] into scalar_result[0]
struct ReduceParams {
    n_workgroups: u32,
}

@group(0) @binding(0) var<storage, read> partial_sums_final: array<f64>;
@group(0) @binding(1) var<storage, read_write> scalar_result: array<f64>;
@group(0) @binding(2) var<uniform> reduce_params: ReduceParams;

var<workgroup> final_shared: array<f64, 256>;

@compute @workgroup_size(256)
fn final_reduce_f64(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let tid = local_id.x;
    let n = reduce_params.n_workgroups;

    // Load partial sums (handle case where n < 256)
    var sum: f64 = 0.0;
    if (tid < n) {
        sum = partial_sums_final[tid];
    }

    // For n > 256, each thread sums multiple elements
    var i = tid + 256u;
    while (i < n) {
        sum = sum + partial_sums_final[i];
        i = i + 256u;
    }

    final_shared[tid] = sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            final_shared[tid] = final_shared[tid] + final_shared[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        scalar_result[0] = final_shared[0];
    }
}
