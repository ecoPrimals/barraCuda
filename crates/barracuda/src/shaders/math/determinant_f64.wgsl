// SPDX-License-Identifier: AGPL-3.0-or-later
// determinant_f64.wgsl - Matrix determinant calculation (f64 canonical)
//
// Computes the determinant of square matrices.
// - 1x1: direct value
// - 2x2: ad - bc formula
// - 3x3: Sarrus rule (cofactor expansion)
// - 4x4: Explicit cofactor expansion along first row
// - NxN (N≤16): LU decomposition with partial pivoting
//
// Each thread processes one matrix in a batch.
// For N>16, use CPU fallback (shader workgroup memory limits).
//
// Cross-domain: Linear algebra, physics (Jacobians), ML (Hessians),
// graphics (transformation matrices).

struct Params {
    matrix_size: u32,    // N for NxN matrix
    total_matrices: u32, // Number of matrices in batch
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;         // Input matrices [batch, N, N]
@group(0) @binding(1) var<storage, read_write> output: array<f64>;  // Output determinants [batch]
@group(0) @binding(2) var<uniform> params: Params;

// 2x2 determinant
fn det_2x2(a: f64, b: f64, c: f64, d: f64) -> f64 {
    return a * d - b * c;
}

// 3x3 determinant via Sarrus rule
fn det_3x3(m: array<f64, 9>) -> f64 {
    let pos = m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7];
    let neg = m[2] * m[4] * m[6] + m[1] * m[3] * m[8] + m[0] * m[5] * m[7];
    return pos - neg;
}

// 4x4 determinant via cofactor expansion along first row
fn det_4x4(offset: u32) -> f64 {
    // Load matrix elements
    var m: array<f64, 16>;
    for (var i: u32 = 0u; i < 16u; i = i + 1u) {
        m[i] = input[offset + i];
    }

    // Cofactor expansion along row 0
    // det = m[0]*M00 - m[1]*M01 + m[2]*M02 - m[3]*M03
    let m00 = m[5] * (m[10] * m[15] - m[11] * m[14])
            - m[6] * (m[9]  * m[15] - m[11] * m[13])
            + m[7] * (m[9]  * m[14] - m[10] * m[13]);

    let m01 = m[4] * (m[10] * m[15] - m[11] * m[14])
            - m[6] * (m[8]  * m[15] - m[11] * m[12])
            + m[7] * (m[8]  * m[14] - m[10] * m[12]);

    let m02 = m[4] * (m[9]  * m[15] - m[11] * m[13])
            - m[5] * (m[8]  * m[15] - m[11] * m[12])
            + m[7] * (m[8]  * m[13] - m[9]  * m[12]);

    let m03 = m[4] * (m[9]  * m[14] - m[10] * m[13])
            - m[5] * (m[8]  * m[14] - m[10] * m[12])
            + m[6] * (m[8]  * m[13] - m[9]  * m[12]);

    return m[0] * m00 - m[1] * m01 + m[2] * m02 - m[3] * m03;
}

// NxN determinant via LU decomposition with partial pivoting (N <= 16)
// Uses per-thread local array (limited by WGSL stack size)
fn det_lu(offset: u32, n: u32) -> f64 {
    // Copy matrix to local working array (max 16x16 = 256 elements)
    var a: array<f64, 256>;
    let n2 = n * n;
    for (var i: u32 = 0u; i < n2; i = i + 1u) {
        a[i] = input[offset + i];
    }

    var sign: f64 = 1.0;

    // Gaussian elimination with partial pivoting
    for (var col: u32 = 0u; col < n; col = col + 1u) {
        // Find pivot (max absolute value in column)
        var max_val: f64 = abs(a[col * n + col]);
        var max_row: u32 = col;
        for (var row: u32 = col + 1u; row < n; row = row + 1u) {
            let val = abs(a[row * n + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Singular check
        if (max_val < 1e-30) {
            return 0.0;
        }

        // Swap rows if needed
        if (max_row != col) {
            sign = -sign;
            for (var j: u32 = 0u; j < n; j = j + 1u) {
                let tmp = a[col * n + j];
                a[col * n + j] = a[max_row * n + j];
                a[max_row * n + j] = tmp;
            }
        }

        // Eliminate below pivot
        let pivot = a[col * n + col];
        for (var row: u32 = col + 1u; row < n; row = row + 1u) {
            let factor = a[row * n + col] / pivot;
            for (var j: u32 = col + 1u; j < n; j = j + 1u) {
                a[row * n + j] = a[row * n + j] - factor * a[col * n + j];
            }
            a[row * n + col] = 0.0;
        }
    }

    // Determinant = sign * product of diagonal
    var det: f64 = sign;
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        det = det * a[i * n + i];
    }
    return det;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let matrix_idx = global_id.x;

    if (matrix_idx >= params.total_matrices) {
        return;
    }

    let n = params.matrix_size;
    let offset = matrix_idx * n * n;

    var det: f64;

    if (n == 1u) {
        det = input[offset];
    } else if (n == 2u) {
        det = det_2x2(
            input[offset], input[offset + 1u],
            input[offset + 2u], input[offset + 3u]
        );
    } else if (n == 3u) {
        var m: array<f64, 9>;
        for (var i: u32 = 0u; i < 9u; i = i + 1u) {
            m[i] = input[offset + i];
        }
        det = det_3x3(m);
    } else if (n == 4u) {
        det = det_4x4(offset);
    } else if (n <= 16u) {
        det = det_lu(offset, n);
    } else {
        // N > 16: too large for per-thread local array
        // Return sentinel value to signal "use CPU fallback"
        det = 3.4028235e+38; // f32::MAX as sentinel
    }

    output[matrix_idx] = det;
}
