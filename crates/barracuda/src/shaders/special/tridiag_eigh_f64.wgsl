// SPDX-License-Identifier: AGPL-3.0-or-later
//
// tridiag_eigh_f64.wgsl — GPU tridiagonal symmetric eigenvector solver (f64)
//
// Implements QL algorithm with implicit Wilkinson shifts for a symmetric
// tridiagonal matrix. Each thread solves one independent tridiagonal system.
//
// Input:  diagonal d[n], sub-diagonal e[n-1] per batch element
// Output: eigenvalues d[n], eigenvectors Z[n×n] per batch element
//
// Absorbed from groundSpring request / wateringHole spectral handoff.
// Dispatch: (n_batches, 1, 1)

enable f64;

struct Params {
    n: u32,
    n_batches: u32,
    max_iter: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> diag: array<f64>;
@group(0) @binding(1) var<storage, read_write> subdiag: array<f64>;
@group(0) @binding(2) var<storage, read_write> eigvecs: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = gid.x;
    if batch >= params.n_batches { return; }

    let n = params.n;
    let d_off = batch * n;
    let e_off = batch * (n - 1u);
    let z_off = batch * n * n;

    // Initialize eigenvectors to identity
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            if i == j {
                eigvecs[z_off + i * n + j] = 1.0;
            } else {
                eigvecs[z_off + i * n + j] = 0.0;
            }
        }
    }

    // QL iteration with implicit Wilkinson shift
    for (var l = 0u; l < n; l = l + 1u) {
        for (var iter = 0u; iter < params.max_iter; iter = iter + 1u) {
            // Find small sub-diagonal element
            var m = l;
            for (var k = l; k < n - 1u; k = k + 1u) {
                let dd = abs(diag[d_off + k]) + abs(diag[d_off + k + 1u]);
                if abs(subdiag[e_off + k]) <= dd * 1e-14 {
                    break;
                }
                m = k + 1u;
            }
            if m == l { break; }

            // Wilkinson shift
            let g = (diag[d_off + l + 1u] - diag[d_off + l]) / (2.0 * subdiag[e_off + l]);
            var r: f64;
            if abs(g) > 1.0 {
                r = abs(g) * sqrt(1.0 + 1.0 / (g * g));
            } else {
                r = sqrt(1.0 + g * g);
            }
            let shift = diag[d_off + m] - diag[d_off + l] + subdiag[e_off + l] / (g + sign(g) * r);

            var s = 1.0;
            var c = 1.0;
            var p = 0.0;

            for (var i = m; i > l; i = i - 1u) {
                let idx = i - 1u;
                let f_val = s * subdiag[e_off + idx];
                let b_val = c * subdiag[e_off + idx];

                if abs(f_val) >= abs(shift) {
                    c = shift / f_val;
                    r = sqrt(c * c + 1.0);
                    subdiag[e_off + idx + 0u] = f_val * r;
                    s = 1.0 / r;
                    c = c * s;
                } else {
                    s = f_val / shift;
                    r = sqrt(s * s + 1.0);
                    subdiag[e_off + idx + 0u] = shift * r;
                    c = 1.0 / r;
                    s = s * c;
                }

                // Intentionally use `idx` which is `i - 1`
                let old_shift = diag[d_off + i] - p;
                let new_r = (diag[d_off + idx] - old_shift) * s + 2.0 * c * b_val;
                p = s * new_r;
                diag[d_off + i] = old_shift + p;

                // Update eigenvectors via Givens rotation
                for (var k = 0u; k < n; k = k + 1u) {
                    let z_ik = eigvecs[z_off + k * n + i];
                    let z_idx = eigvecs[z_off + k * n + idx];
                    eigvecs[z_off + k * n + i] = s * z_idx + c * z_ik;
                    eigvecs[z_off + k * n + idx] = c * z_idx - s * z_ik;
                }

                if idx == l { break; }
            }

            diag[d_off + l] = diag[d_off + l] - p;
            subdiag[e_off + l] = shift;
            if m < n - 1u {
                subdiag[e_off + m] = 0.0;
            }
        }
    }
}
