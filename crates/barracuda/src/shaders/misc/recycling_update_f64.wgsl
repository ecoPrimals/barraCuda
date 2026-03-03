// SPDX-License-Identifier: AGPL-3.0-or-later
//
// recycling_update_f64.wgsl — AlphaFold2 recycling iteration update
//
// out = prev + layer_norm(current - prev)
// Uses running difference with layer norm for iterative refinement.
//
// Bindings: @0 prev[N*C], @1 current[N*C], @2 out[N*C], @3 uniform{n, c, eps}

enable f64;

struct RecyclingParams {
    n: u32,
    c: u32,
    _pad: [u32; 2],
    eps: f64,
}

@group(0) @binding(0) var<storage, read>       prev: array<f64>;    // [N*C]
@group(0) @binding(1) var<storage, read>       current: array<f64>; // [N*C]
@group(0) @binding(2) var<storage, read_write> out: array<f64>;   // [N*C]
@group(0) @binding(3) var<uniform>             params: RecyclingParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n;
    let C = params.c;
    let eps = params.eps;

    let idx = gid.x;
    if idx >= N * C { return; }

    let row = idx / C;
    let col = idx % C;

    // Compute diff = current - prev for this row (need full row for layer norm)
    // Each thread handles one element; we need shared work for layer norm.
    // Simplified: each thread computes its own diff, then we need a second pass for layer norm.
    // For per-row layer norm, we need to compute mean and var over C elements for row.
    // Use a loop to compute mean and var for this row (each thread does full row - not ideal but works)
    let row_base = row * C;

    var mean_val = f64(0.0);
    for (var k = 0u; k < C; k = k + 1u) {
        mean_val += current[row_base + k] - prev[row_base + k];
    }
    mean_val /= f64(C);

    var var_val = f64(0.0);
    for (var k = 0u; k < C; k = k + 1u) {
        let d = (current[row_base + k] - prev[row_base + k]) - mean_val;
        var_val += d * d;
    }
    var_val = sqrt_f64(var_val / f64(C) + eps);

    let diff = (current[idx] - prev[idx]) - mean_val;
    let normalized = select(f64(0.0), diff / var_val, var_val > eps);

    out[idx] = prev[idx] + normalized;
}
