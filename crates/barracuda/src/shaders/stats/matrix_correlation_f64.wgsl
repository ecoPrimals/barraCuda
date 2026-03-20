// SPDX-License-Identifier: AGPL-3.0-or-later
//
// matrix_correlation_f64.wgsl — Pairwise Pearson correlation matrix
//
// For P columns of N observations: corr[i,j] = cov(i,j) / (std(i) * std(j))
// Each workgroup item computes one (i,j) entry of the P*P correlation matrix.
//
// Bindings: @0 data[N*P], @1 out[P*P], @2 uniform{n: u32, p: u32}
//
// Provenance: neuralSpring → ToadStool absorption

enable f64;

struct CorrParams {
    n: u32,
    p: u32,
}

@group(0) @binding(0) var<storage, read>       data: array<f64>;   // [N*P] row-major
@group(0) @binding(1) var<storage, read_write> out:  array<f64>;   // [P*P]
@group(0) @binding(2) var<uniform>             params: CorrParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let P = params.p;
    let N = params.n;

    if idx >= P * P { return; }

    let i = idx / P;
    let j = idx % P;

    var sum_i: f64 = 0.0;
    var sum_j: f64 = 0.0;
    var sum_ii: f64 = 0.0;
    var sum_jj: f64 = 0.0;
    var sum_ij: f64 = 0.0;

    for (var k: u32 = 0u; k < N; k = k + 1u) {
        let vi = data[k * P + i];
        let vj = data[k * P + j];
        sum_i += vi;
        sum_j += vj;
        sum_ii += vi * vi;
        sum_jj += vj * vj;
        sum_ij += vi * vj;
    }

    let n = f64(N);
    let mean_i = sum_i / n;
    let mean_j = sum_j / n;

    let cov_ij = sum_ij / n - mean_i * mean_j;
    let var_i = sum_ii / n - mean_i * mean_i;
    let var_j = sum_jj / n - mean_j * mean_j;

    let denom = sqrt_f64(var_i * var_j);
    if denom > 1.0e-15 {
        out[idx] = cov_ij / denom;
    } else {
        out[idx] = select(f64(0.0), f64(1.0), i == j);
    }
}
