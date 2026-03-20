// SPDX-License-Identifier: AGPL-3.0-or-later
// Variance reduction - Var(X) = E[(X - μ)²] (f64 canonical)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> shared_var: f64;

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
        shared_var = variance / f64(size);
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        output[0] = shared_var;
    }
}
