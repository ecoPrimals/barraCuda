// SPDX-License-Identifier: AGPL-3.0-or-later
// Element-wise addition: C = A + B (f64 - double precision)
//
// **Purpose**: High-precision element-wise addition for scientific computing
// **Precision**: f64 (IEEE 754 double precision, ~15 decimal digits)
//
// **When to use fp64**:
// ✅ Scientific computing requiring high precision
// ✅ Numerical stability in ill-conditioned problems
// ✅ Accumulation across many iterations (gradient sums)
// ❌ Real-time inference (use fp32)
//
// **Performance**:
// Consumer GPUs: 1:32 (NVIDIA) to 1:16 (AMD) of fp32 throughput
// Workstation GPUs (Titan V, A100): 1:2 of fp32 throughput
//
// **Requirements**: Device must support SHADER_F64 feature (most Vulkan GPUs do)

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    // C = A + B (double precision)
    output[idx] = a[idx] + b[idx];
}
