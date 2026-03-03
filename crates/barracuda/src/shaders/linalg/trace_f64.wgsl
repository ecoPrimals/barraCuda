// Trace - Sum of diagonal elements - Pure GPU Tree Reduction (f64 canonical)
// Computes sum of diagonal elements: tr(A) = Σ A[i,i]
//
// Algorithm:
// 1. Each thread reads a diagonal element: A[i,i]
// 2. Tree reduction in workgroup shared memory
// 3. Single workgroup writes final sum to output[0]
//
// For matrices larger than 256, uses multiple workgroups with partial results
// then reduces partial results in a second pass if needed.

struct Params {
    size: u32,       // Matrix size (n×n)
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> matrix: array<f64>;         // [size, size]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;   // [1] or partial results
@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load diagonal element into shared memory
    var value: f64 = f64(0.0);
    if (gid < params.size) {
        // Read diagonal element: matrix[i * size + i]
        value = matrix[gid * params.size + gid];
    }
    shared_data[tid] = value;
    workgroupBarrier();
    
    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            shared_data[tid] = a + b;
        }
        workgroupBarrier();
    }
    
    // Write result (single workgroup case) or partial result (multi-workgroup case)
    if (tid == 0u) {
        if (params.size <= 256u) {
            // Single workgroup: write final result
            output[0] = shared_data[0];
        } else {
            // Multiple workgroups: write partial result
            output[workgroup_id.x] = shared_data[0];
        }
    }
}
