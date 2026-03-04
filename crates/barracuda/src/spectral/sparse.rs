// SPDX-License-Identifier: AGPL-3.0-or-later

//! Compressed Sparse Row matrix format for spectral theory.
//!
//! Simplified CSR for symmetric square matrices (common in spectral theory).
//! For general m×n sparse matrices, use `crate::linalg::sparse::CsrMatrix`.

/// Sparse symmetric matrix in Compressed Sparse Row format for spectral theory.
///
/// This is a specialized format for n×n symmetric matrices common in
/// discrete Schrödinger operators. For general sparse matrices, use
/// `crate::linalg::sparse::CsrMatrix`.
#[derive(Debug, Clone)]
pub struct SpectralCsrMatrix {
    /// Matrix dimension (n × n)
    pub n: usize,
    /// Row pointers (length n+1)
    pub row_ptr: Vec<usize>,
    /// Column indices (length nnz)
    pub col_idx: Vec<usize>,
    /// Non-zero values (length nnz)
    pub values: Vec<f64>,
}

impl SpectralCsrMatrix {
    /// Sparse matrix-vector product: y = A * x.
    ///
    /// This is the P1 primitive for GPU promotion — the inner loop of Lanczos.
    /// CPU version; GPU SpMV available via `WGSL_SPMV_CSR_F64` shader.
    pub fn spmv(&self, x: &[f64], y: &mut [f64]) {
        for (i, yi) in y.iter_mut().enumerate().take(self.n) {
            let mut sum = 0.0;
            for j in self.row_ptr[i]..self.row_ptr[i + 1] {
                sum += self.values[j] * x[self.col_idx[j]];
            }
            *yi = sum;
        }
    }

    /// Number of non-zero entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

/// WGSL compute shader for CSR sparse matrix-vector product y = A*x (f64).
///
/// Direct GPU port of [`SpectralCsrMatrix::spmv()`]: one thread per matrix row,
/// f64 multiply-accumulate in the inner loop.
///
/// ## Binding layout
///
/// | Binding | Type | Content |
/// |---------|------|---------|
/// | 0 | uniform | `{ n: u32, nnz: u32, pad: u32, pad: u32 }` |
/// | 1 | storage, read | `row_ptr: array<u32>` (n+1 entries) |
/// | 2 | storage, read | `col_idx: array<u32>` (nnz entries) |
/// | 3 | storage, read | `values: array<f64>` (nnz entries) |
/// | 4 | storage, read | `x: array<f64>` (n entries, input) |
/// | 5 | storage, read_write | `y: array<f64>` (n entries, output) |
///
/// ## Dispatch
///
/// `ceil(n / 64)` workgroups of 64 threads.
///
/// ## Provenance
///
/// GPU promotion of CPU SpMV for Kachkovskiy spectral theory GPU Lanczos
/// and lattice QCD GPU Dirac. Absorbed from hotSpring v0.6.0.
pub const WGSL_SPMV_CSR_F64: &str = r"
struct Params {
    n: u32,
    nnz: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> row_ptr: array<u32>;
@group(0) @binding(2) var<storage, read> col_idx: array<u32>;
@group(0) @binding(3) var<storage, read> vals: array<f64>;
@group(0) @binding(4) var<storage, read> x_vec: array<f64>;
@group(0) @binding(5) var<storage, read_write> y_vec: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.n {
        return;
    }

    let start = row_ptr[row];
    let end = row_ptr[row + 1u];

    var sum: f64 = f64(0.0);
    for (var j = start; j < end; j = j + 1u) {
        sum = sum + vals[j] * x_vec[col_idx[j]];
    }

    y_vec[row] = sum;
}
";

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn csr_spmv_identity() {
        let mat = SpectralCsrMatrix {
            n: 3,
            row_ptr: vec![0, 1, 2, 3],
            col_idx: vec![0, 1, 2],
            values: vec![1.0, 1.0, 1.0],
        };
        let x = vec![3.0, 5.0, 7.0];
        let mut y = vec![0.0; 3];
        mat.spmv(&x, &mut y);
        assert!((y[0] - 3.0).abs() < 1e-14);
        assert!((y[1] - 5.0).abs() < 1e-14);
        assert!((y[2] - 7.0).abs() < 1e-14);
    }

    #[test]
    fn csr_spmv_tridiag() {
        let mat = SpectralCsrMatrix {
            n: 4,
            row_ptr: vec![0, 2, 5, 8, 10],
            col_idx: vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3],
            values: vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        };
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let mut y = vec![0.0; 4];
        mat.spmv(&x, &mut y);
        assert!((y[0] - 2.0).abs() < 1e-14);
        assert!((y[1] - -1.0).abs() < 1e-14);
        assert!((y[2] - 0.0).abs() < 1e-14);
        assert!((y[3] - 0.0).abs() < 1e-14);
    }
}
