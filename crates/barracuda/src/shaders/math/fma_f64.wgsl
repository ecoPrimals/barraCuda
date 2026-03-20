// SPDX-License-Identifier: AGPL-3.0-or-later
// Fused Multiply-Add: D = A * B + C (f64 canonical)
// Single memory pass instead of mul then add (2 passes)
// This is a key optimization for patterns like:
//   - Linear layers: output = weight @ input + bias
//   - Residual connections: output = layer(x) + x
//   - Scaled additions: output = alpha * x + y

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read> c: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    // D = A * B + C
    // WGSL fma() is not defined for f64; Sovereign Compiler fuses a*b+c to SPIR-V FMA
    output[idx] = a[idx] * b[idx] + c[idx];
}
