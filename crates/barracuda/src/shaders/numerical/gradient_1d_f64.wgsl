// SPDX-License-Identifier: AGPL-3.0-or-later
//
// gradient_1d_f64.wgsl — Parallel 1D numerical gradient (numpy.gradient compatible)
//
// Interior: 2nd-order central difference (f[i+1] - f[i-1]) / (2·dx)
// Boundary: 2nd-order forward/backward stencils
//
// Bindings: @0 f[N], @1 out[N], @2 uniform { n: u32, dx: f64 }

struct Params {
    n: u32,
    _pad: u32,
    dx: f64,
}

@group(0) @binding(0) var<storage, read> f: array<f64>;
@group(0) @binding(1) var<storage, read_write> out: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if i >= n { return; }
    let dx = params.dx;

    if n == 1u {
        out[0u] = 0.0;
        return;
    }

    if n == 2u {
        out[i] = (f[1u] - f[0u]) / dx;
        return;
    }

    if i == 0u {
        // 2nd-order forward: (-3f[0] + 4f[1] - f[2]) / (2dx)
        out[0u] = (-3.0 * f[0u] + 4.0 * f[1u] - f[2u]) / (2.0 * dx);
    } else if i == n - 1u {
        // 2nd-order backward: (3f[n-1] - 4f[n-2] + f[n-3]) / (2dx)
        out[i] = (3.0 * f[i] - 4.0 * f[i - 1u] + f[i - 2u]) / (2.0 * dx);
    } else {
        // Central difference: (f[i+1] - f[i-1]) / (2dx)
        out[i] = (f[i + 1u] - f[i - 1u]) / (2.0 * dx);
    }
}
