// Standard deviation - Std(X) = sqrt(Var(X)) (f64 canonical)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> shared_std: f64;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = arrayLength(&input);
    let tid = local_id.x;
    
    if (tid == 0u) {
        // Compute mean
        var sum = 0.0;
        for (var i = 0u; i < size; i = i + 1u) {
            sum = sum + input[i];
        }
        let mean = sum / f64(size);
        
        // Compute variance
        var variance = 0.0;
        for (var i = 0u; i < size; i = i + 1u) {
            let diff = input[i] - mean;
            variance = variance + diff * diff;
        }
        variance = variance / f64(size);
        
        // Standard deviation is sqrt of variance
        shared_std = sqrt_f64(variance);
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        output[0] = shared_std;
    }
}
