// SPDX-License-Identifier: AGPL-3.0-or-later
// Fused Multiply-Add: D = A * B + C
// AMD-optimized: WG=128 (benchmarks show 2x faster than WG=64)

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(128)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    
    if (idx >= arrayLength(&output)) {
        return;
    }
    
    output[idx] = fma(a[idx], b[idx], c[idx]);
}
