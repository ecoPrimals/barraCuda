// Reduce: Compute sum/max/min/mean over input (f64 canonical)
// CUDA equivalent: thrust::reduce, cub::DeviceReduce
// Algorithm: Tree reduction (work-efficient)
// Use cases: Loss computation, gradient accumulation

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // Partial results

struct Params {
    size: u32,
    operation: u32,  // 0=Sum, 1=Max, 2=Min, 3=Mean
    _pad0: u32,
    _pad1: u32,
}
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
    
    // Load data into shared memory
    var value: f64;
    if (gid < params.size) {
        value = input[gid];
    } else {
        // Initialize with identity based on operation
        if (params.operation == 0u || params.operation == 3u) {  // Sum, Mean
            value = 0.0;
        } else if (params.operation == 1u) {  // Max
            value = -1.7976931348623157e+308;  // -DBL_MAX
        } else if (params.operation == 2u) {  // Min
            value = 1.7976931348623157e+308;   // DBL_MAX
        } else {
            value = 0.0;
        }
    }
    shared_data[tid] = value;
    workgroupBarrier();
    
    // Tree reduction in shared memory
    // Use exact same pattern as sum_reduce.wgsl which is known to work
    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            let a = shared_data[tid];
            let b = shared_data[tid + stride];
            
            var result: f64;
            if (params.operation == 0u || params.operation == 3u) {  // Sum, Mean
                result = a + b;
            } else if (params.operation == 1u) {  // Max
                result = max(a, b);
            } else if (params.operation == 2u) {  // Min
                result = min(a, b);
            } else {
                result = a;
            }
            shared_data[tid] = result;
        }
        workgroupBarrier();
    }
    
    // Thread 0 writes workgroup result
    if (tid == 0u) {
        var result = shared_data[0];
        
        // Deep Debt Fix: Implement Mean operation (divide by size)
        if (params.operation == 3u) {
            result = result / f64(params.size);
        }
        
        output[workgroup_id.x] = result;
    }
}
