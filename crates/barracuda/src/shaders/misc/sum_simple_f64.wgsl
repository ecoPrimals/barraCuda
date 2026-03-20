// SPDX-License-Identifier: AGPL-3.0-or-later
// Sum reduction - Simple single-workgroup version
// Computes sum of all elements

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> shared_sum: f64;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = arrayLength(&input);
    let tid = local_id.x;
    
    // Phase 1: Sum all elements (serial for simplicity)
    if (tid == 0u) {
        var sum = 0.0;
        for (var i = 0u; i < size; i = i + 1u) {
            sum = sum + input[i];
        }
        shared_sum = sum;
    }
    workgroupBarrier();
    
    // Write result
    if (tid == 0u) {
        output[0] = shared_sum;
    }
}
