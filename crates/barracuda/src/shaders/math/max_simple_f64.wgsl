// SPDX-License-Identifier: AGPL-3.0-or-later
// Max reduction - Find maximum value (f64 canonical)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> shared_max: f64;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = arrayLength(&input);
    let tid = local_id.x;
    
    if (tid == 0u) {
        var max_val = input[0];
        for (var i = 1u; i < size; i = i + 1u) {
            max_val = max(max_val, input[i]);
        }
        shared_max = max_val;
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        output[0] = shared_max;
    }
}
