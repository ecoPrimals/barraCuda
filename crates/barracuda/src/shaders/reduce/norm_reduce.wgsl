// Norm Reduction: Compute p-norm over all elements
// Formula: (sum(|x|^p))^(1/p)
// CUDA equivalent: thrust::reduce with norm operation
// Algorithm: Tree reduction (work-efficient)
// Use cases: Global norm computation

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;  // Partial results

struct Params {
    size: u32,
    p: f32,  // p-norm parameter
}

@group(0) @binding(2) var<uniform> params: Params;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data into shared memory
    var value: f32;
    if (gid < params.size) {
        let x = input[gid];
        // Compute |x|^p
        value = pow(abs(x), params.p);
    } else {
        // Initialize with 0 for sum reduction
        value = 0.0;
    }
    shared_data[tid] = value;
    workgroupBarrier();
    
    // Tree reduction in shared memory
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride && (gid + stride) < params.size) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            shared_data[tid] = a + b;
        }
        workgroupBarrier();
    }
    
    // Write partial result (sum of |x|^p, not final norm - root happens on CPU)
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
