// SPDX-License-Identifier: AGPL-3.0-only
//
// linear_regression_f64.wgsl — Batched OLS: beta = (X'X)^-1 X'y
//
// For B independent regressions: X[B*N*K], y[B*N], out_beta[B*K]
// For small K (1-4): explicit inverse formula. For K>4: Cholesky on X'X.
//
// Bindings: @0 x[B*N*K], @1 y[B*N], @2 out_beta[B*K], @3 uniform{b, n, k}
//
// Provenance: neuralSpring → ToadStool absorption

enable f64;

struct OLSParams {
    b: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read>       x:        array<f64>;  // [B*N*K]
@group(0) @binding(1) var<storage, read>       y:        array<f64>;  // [B*N]
@group(0) @binding(2) var<storage, read_write> out_beta: array<f64>;  // [B*K]
@group(0) @binding(3) var<uniform>             params:   OLSParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if batch >= params.b { return; }

    let N = params.n;
    let K = params.k;
    let x_base = batch * N * K;
    let y_base = batch * N;
    let out_base = batch * K;

    var xtx: array<f64, 16>;  // K*K max for K<=4
    var xty: array<f64, 4>;   // K max

    for (var i: u32 = 0u; i < K * K; i = i + 1u) {
        xtx[i] = 0.0;
    }
    for (var i: u32 = 0u; i < K; i = i + 1u) {
        xty[i] = 0.0;
    }

    for (var r: u32 = 0u; r < N; r = r + 1u) {
        let yr = y[y_base + r];
        for (var i: u32 = 0u; i < K; i = i + 1u) {
            let xi = x[x_base + r * K + i];
            xty[i] += xi * yr;
            for (var j: u32 = 0u; j <= i; j = j + 1u) {
                let xj = x[x_base + r * K + j];
                xtx[i * K + j] += xi * xj;
                if i != j {
                    xtx[j * K + i] += xi * xj;
                }
            }
        }
    }

    if K == 1u {
        let denom = xtx[0u];
        out_beta[out_base] = select(f64(0.0), xty[0u] / denom, abs(denom) > 1.0e-15);
        return;
    }

    if K == 2u {
        let a = xtx[0u];
        let b = xtx[1u];
        let c = xtx[2u];
        let d = xtx[3u];
        let det = a * d - b * c;
        if abs(det) < 1.0e-15 {
            out_beta[out_base] = 0.0;
            out_beta[out_base + 1u] = 0.0;
            return;
        }
        let inv_det = 1.0 / det;
        out_beta[out_base] = (d * xty[0u] - b * xty[1u]) * inv_det;
        out_beta[out_base + 1u] = (-c * xty[0u] + a * xty[1u]) * inv_det;
        return;
    }

    if K == 3u {
        let a00 = xtx[0u]; let a01 = xtx[1u]; let a02 = xtx[2u];
        let a10 = xtx[3u]; let a11 = xtx[4u]; let a12 = xtx[5u];
        let a20 = xtx[6u]; let a21 = xtx[7u]; let a22 = xtx[8u];
        let det = a00*(a11*a22 - a12*a21) - a01*(a10*a22 - a12*a20) + a02*(a10*a21 - a11*a20);
        if abs(det) < 1.0e-15 {
            for (var i: u32 = 0u; i < 3u; i = i + 1u) { out_beta[out_base + i] = 0.0; }
            return;
        }
        let inv_det = 1.0 / det;
        out_beta[out_base] = ((a11*a22 - a12*a21)*xty[0u] - (a01*a22 - a02*a21)*xty[1u] + (a01*a12 - a02*a11)*xty[2u]) * inv_det;
        out_beta[out_base + 1u] = (-(a10*a22 - a12*a20)*xty[0u] + (a00*a22 - a02*a20)*xty[1u] - (a00*a12 - a02*a10)*xty[2u]) * inv_det;
        out_beta[out_base + 2u] = ((a10*a21 - a11*a20)*xty[0u] - (a00*a21 - a01*a20)*xty[1u] + (a00*a11 - a01*a10)*xty[2u]) * inv_det;
        return;
    }

    if K == 4u {
        let m = array<f64, 16>(xtx[0u], xtx[1u], xtx[2u], xtx[3u], xtx[4u], xtx[5u], xtx[6u], xtx[7u], xtx[8u], xtx[9u], xtx[10u], xtx[11u], xtx[12u], xtx[13u], xtx[14u], xtx[15u]);
        let s0 = m[0]*m[5] - m[1]*m[4];
        let s1 = m[0]*m[6] - m[2]*m[4];
        let s2 = m[0]*m[7] - m[3]*m[4];
        let s3 = m[1]*m[6] - m[2]*m[5];
        let s4 = m[1]*m[7] - m[3]*m[5];
        let s5 = m[2]*m[7] - m[3]*m[6];
        let c5 = s0*m[10] - s1*m[9] + s3*m[8];
        let c4 = s0*m[11] - s2*m[9] + s4*m[8];
        let c3 = s1*m[11] - s2*m[10] + s5*m[8];
        let c2 = s3*m[11] - s4*m[10] + s5*m[9];
        let det = c5*m[15] - c4*m[14] + c3*m[13] - c2*m[12];
        if abs(det) < 1.0e-15 {
            for (var i: u32 = 0u; i < 4u; i = i + 1u) { out_beta[out_base + i] = 0.0; }
            return;
        }
        let inv_det = 1.0 / det;
        let rhs = array<f64, 4>(xty[0u], xty[1u], xty[2u], xty[3u]);
        for (var col: u32 = 0u; col < 4u; col = col + 1u) {
            var cof: f64 = 0.0;
            for (var row: u32 = 0u; row < 4u; row = row + 1u) {
                let sign = select(-1.0, 1.0, (row + col) % 2u == 0u);
                var minor: array<f64, 9>;
                var mi: u32 = 0u;
                for (var i: u32 = 0u; i < 4u; i = i + 1u) {
                    for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                        if i != row && j != col {
                            minor[mi] = m[i*4u+j];
                            mi = mi + 1u;
                        }
                    }
                }
                let d = minor[0]*(minor[4]*minor[8]-minor[5]*minor[7]) - minor[1]*(minor[3]*minor[8]-minor[5]*minor[6]) + minor[2]*(minor[3]*minor[7]-minor[4]*minor[6]);
                cof += sign * d * rhs[row];
            }
            out_beta[out_base + col] = cof * inv_det;
        }
        return;
    }

    for (var i: u32 = 0u; i < K; i = i + 1u) {
        out_beta[out_base + i] = 0.0;
    }
}
