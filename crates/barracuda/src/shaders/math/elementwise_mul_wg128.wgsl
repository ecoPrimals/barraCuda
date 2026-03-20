// SPDX-License-Identifier: AGPL-3.0-or-later
// Optimized element-wise multiplication for AMD GPUs
// Workgroup size 128 provides best performance on AMD RADV driver

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] = a[idx] * b[idx];
}
