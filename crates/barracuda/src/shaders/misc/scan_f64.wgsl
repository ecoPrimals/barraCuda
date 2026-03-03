// Scan (Prefix Sum): Work-efficient parallel scan (Blelloch algorithm) (f64 canonical)
// CUDA equivalent: thrust::scan, cub::DeviceScan
// Use cases: Cumulative sums, stream compaction, allocation

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

struct Params {
    size: u32,
    operation: u32,  // 0=Sum, 1=Max, 2=Min
    exclusive: u32,  // 0=inclusive, 1=exclusive
}
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> temp: array<f64, 512>;  // Double-buffered for up-sweep and down-sweep

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x * 2u;
    
    // Load input into shared memory
    if (gid < params.size) {
        temp[tid * 2u] = input[gid];
    } else {
        temp[tid * 2u] = 0.0;
    }
    
    if (gid + 1u < params.size) {
        temp[tid * 2u + 1u] = input[gid + 1u];
    } else {
        temp[tid * 2u + 1u] = 0.0;
    }
    workgroupBarrier();
    
    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = 256u; d > 0u; d = d / 2u) {
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            temp[bi] = temp[bi] + temp[ai];
        }
        offset = offset * 2u;
    }
    
    // Store the total sum for inclusive scan conversion
    var total_sum: f64;
    if (tid == 0u) {
        total_sum = temp[511];  // Save total before clearing
        temp[511] = 0.0;  // Always clear for exclusive scan
    }
    workgroupBarrier();
    
    // Down-sweep phase (produces exclusive scan)
    for (var d = 1u; d <= 256u; d = d * 2u) {
        offset = offset / 2u;
        workgroupBarrier();
        if (tid < d) {
            let ai = offset * (2u * tid + 1u) - 1u;
            let bi = offset * (2u * tid + 2u) - 1u;
            
            let t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] = temp[bi] + t;
        }
    }
    workgroupBarrier();
    
    // Write results
    // EVOLUTION FIX: For inclusive scan, add original input to exclusive result
    if (params.exclusive == 0u) {
        // Inclusive: each element includes itself
        if (gid < params.size) {
            output[gid] = temp[tid * 2u] + input[gid];
        }
        if (gid + 1u < params.size) {
            output[gid + 1u] = temp[tid * 2u + 1u] + input[gid + 1u];
        }
    } else {
        // Exclusive: each element is sum of previous elements only
        if (gid < params.size) {
            output[gid] = temp[tid * 2u];
        }
        if (gid + 1u < params.size) {
            output[gid + 1u] = temp[tid * 2u + 1u];
        }
    }
}
