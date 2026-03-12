// SPDX-License-Identifier: AGPL-3.0-only
// Real-to-complex zero-interleave — f64 canonical.
// Converts a real signal [N] to complex [N, 2] by appending zero imaginary parts.
// coralReef handles precision lowering; barraCuda emits max precision.

@group(0) @binding(0) var<storage, read>       real_input:     array<f64>;
@group(0) @binding(1) var<storage, read_write> complex_output: array<f64>;
@group(0) @binding(2) var<uniform>             params:         RtcParams;

struct RtcParams {
    n:    u32,
    _p1:  u32,
    _p2:  u32,
    _p3:  u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n { return; }
    complex_output[idx * 2u]       = real_input[idx];
    complex_output[idx * 2u + 1u]  = f64(0);
}
