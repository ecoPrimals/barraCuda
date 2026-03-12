// SPDX-License-Identifier: AGPL-3.0-only
// L2 Norm - ||x|| = sqrt(sum(x²)) (f64 canonical)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

var<workgroup> shared_norm: f64;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>) {
    let size = arrayLength(&input);
    let tid = local_id.x;
    
    if (tid == 0u) {
        var sum_squares = 0.0;
        for (var i = 0u; i < size; i = i + 1u) {
            let val = input[i];
            sum_squares = sum_squares + val * val;
        }
        shared_norm = sqrt_f64(sum_squares);
    }
    workgroupBarrier();
    
    if (tid == 0u) {
        output[0] = shared_norm;
    }
}
