// SPDX-License-Identifier: AGPL-3.0-only
// Optimized element-wise addition for AMD GPUs
// Workgroup size 128 provides best performance on AMD RADV driver
// (2 wavefronts of 64 threads each)
//
// Benchmark: WG=128 is 2x faster than WG=64 on RX 6950 XT

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] = a[idx] + b[idx];
}
