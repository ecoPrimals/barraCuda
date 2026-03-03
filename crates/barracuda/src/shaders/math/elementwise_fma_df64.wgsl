// SPDX-License-Identifier: AGPL-3.0-or-later
// Elementwise fused multiply-add at DF64 precision (f32-pair, ~48-bit mantissa).
// result[i] = a[i] * b[i] + c[i]
// Input/output as vec2<f32> where .x = hi, .y = lo.
// Requires: df64_core.wgsl (auto-injected by compile_shader_df64)

@group(0) @binding(0) var<storage, read> a: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> c: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> result: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> n: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= n { return; }
    let va = Df64(a[idx].x, a[idx].y);
    let vb = Df64(b[idx].x, b[idx].y);
    let vc = Df64(c[idx].x, c[idx].y);
    let r = df64_add(df64_mul(va, vb), vc);
    result[idx] = vec2<f32>(r.hi, r.lo);
}
