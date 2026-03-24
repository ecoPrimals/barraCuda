// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compressed Sparse Row (CSR) matrix format
//!
//! CSR is the standard format for sparse matrix operations:
//! - Efficient row access O(1)
//! - Efficient `SpMV` O(nnz)
//! - Memory: O(nnz + `n_rows`)
//!
//! # Format
//!
//! A CSR matrix stores:
//! - `values`: Non-zero values (length = nnz)
//! - `col_indices`: Column index for each value (length = nnz)
//! - `row_ptr`: Start of each row in `values/col_indices` (length = `n_rows` + 1)
//!
//! # Example
//!
//! ```text
//! Matrix:     CSR:
//! [1 0 2]     values = [1, 2, 3, 4, 5]
//! [0 3 0]     col_indices = [0, 2, 1, 0, 2]
//! [4 0 5]     row_ptr = [0, 2, 3, 5]
//! ```

use crate::error::{BarracudaError, Result};

/// Coordinate (COO) format - easy construction
#[derive(Debug, Clone)]
pub struct CooMatrix {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Row indices
    pub row_indices: Vec<usize>,
    /// Column indices
    pub col_indices: Vec<usize>,
    /// Values
    pub values: Vec<f64>,
}

impl CooMatrix {
    /// Create a new empty COO matrix
    #[must_use]
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Create from triplets (row, col, value)
    #[must_use]
    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let mut mat = Self::new(n_rows, n_cols);
        for &(row, col, val) in triplets {
            mat.add(row, col, val);
        }
        mat
    }

    /// Add a value at (row, col)
    pub fn add(&mut self, row: usize, col: usize, value: f64) {
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(value);
    }

    /// Number of non-zeros
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Convert to CSR format
    #[must_use]
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_coo(self)
    }
}

/// Compressed Sparse Row (CSR) matrix format
#[derive(Debug, Clone)]
pub struct CsrMatrix {
    /// Number of rows
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
    /// Non-zero values (length = nnz)
    pub values: Vec<f64>,
    /// Column indices (length = nnz)
    pub col_indices: Vec<usize>,
    /// Row pointers (length = `n_rows` + 1)
    pub row_ptr: Vec<usize>,
}

impl CsrMatrix {
    /// Create an empty CSR matrix
    #[must_use]
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            n_rows,
            n_cols,
            values: Vec::new(),
            col_indices: Vec::new(),
            row_ptr: vec![0; n_rows + 1],
        }
    }

    /// Create from triplets (row, col, value)
    #[must_use]
    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(usize, usize, f64)]) -> Self {
        let coo = CooMatrix::from_triplets(n_rows, n_cols, triplets);
        Self::from_coo(&coo)
    }

    /// Create from COO format
    #[must_use]
    pub fn from_coo(coo: &CooMatrix) -> Self {
        let n_rows = coo.n_rows;
        let n_cols = coo.n_cols;
        let nnz = coo.nnz();

        if nnz == 0 {
            return Self::new(n_rows, n_cols);
        }

        // Sort by row, then column
        let mut indices: Vec<usize> = (0..nnz).collect();
        indices.sort_by_key(|&i| (coo.row_indices[i], coo.col_indices[i]));

        // Build CSR arrays
        let mut values = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut row_ptr = vec![0; n_rows + 1];

        let mut current_row = 0;
        for &idx in &indices {
            let row = coo.row_indices[idx];

            // Fill row_ptr for any empty rows
            while current_row < row {
                current_row += 1;
                row_ptr[current_row] = values.len();
            }

            values.push(coo.values[idx]);
            col_indices.push(coo.col_indices[idx]);
        }

        // Fill remaining row_ptr entries
        while current_row < n_rows {
            current_row += 1;
            row_ptr[current_row] = values.len();
        }

        Self {
            n_rows,
            n_cols,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Create from triplets, summing duplicate `(row, col)` entries.
    ///
    /// Scientific codes often accumulate element contributions where the same
    /// `(i, j)` position receives multiple values that should be added. This
    /// builder handles that automatically, unlike [`from_triplets`](Self::from_triplets)
    /// which preserves duplicates as separate entries.
    ///
    /// # Example
    ///
    /// ```
    /// # use barracuda::linalg::sparse::CsrMatrix;
    /// let csr = CsrMatrix::from_triplets_summed(2, 2, &[
    ///     (0, 0, 1.0),
    ///     (0, 0, 2.0),  // duplicate — will be summed
    ///     (1, 1, 3.0),
    /// ]);
    /// assert!((csr.get(0, 0) - 3.0).abs() < 1e-10);
    /// ```
    #[must_use]
    pub fn from_triplets_summed(
        n_rows: usize,
        n_cols: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Self {
        if triplets.is_empty() {
            return Self::new(n_rows, n_cols);
        }

        let mut sorted: Vec<(usize, usize, f64)> = triplets.to_vec();
        sorted.sort_by_key(|&(r, c, _)| (r, c));

        let mut values = Vec::with_capacity(triplets.len());
        let mut col_indices = Vec::with_capacity(triplets.len());
        let mut row_ptr = vec![0usize; n_rows + 1];

        let mut prev_row = sorted[0].0;
        let mut prev_col = sorted[0].1;
        let mut acc = sorted[0].2;

        for &(r, c, v) in &sorted[1..] {
            if r == prev_row && c == prev_col {
                acc += v;
            } else {
                for ptr in &mut row_ptr[prev_row + 1..] {
                    *ptr = values.len() + 1;
                }
                values.push(acc);
                col_indices.push(prev_col);
                prev_row = r;
                prev_col = c;
                acc = v;
            }
        }
        values.push(acc);
        col_indices.push(prev_col);
        for ptr in &mut row_ptr[prev_row + 1..] {
            *ptr = values.len();
        }

        Self {
            n_rows,
            n_cols,
            values,
            col_indices,
            row_ptr,
        }
    }

    /// Create identity matrix
    #[must_use]
    pub fn identity(n: usize) -> Self {
        let triplets: Vec<_> = (0..n).map(|i| (i, i, 1.0)).collect();
        Self::from_triplets(n, n, &triplets)
    }

    /// Create diagonal matrix from values
    #[must_use]
    pub fn from_diagonal(diag: &[f64]) -> Self {
        let n = diag.len();
        let triplets: Vec<_> = diag.iter().enumerate().map(|(i, &v)| (i, i, v)).collect();
        Self::from_triplets(n, n, &triplets)
    }

    /// Create tridiagonal matrix
    /// # Errors
    /// Returns [`Err`] if lower or upper diagonal length does not equal `main.len() - 1`.
    pub fn tridiagonal(lower: &[f64], main: &[f64], upper: &[f64]) -> Result<Self> {
        let n = main.len();
        if lower.len() != n - 1 || upper.len() != n - 1 {
            return Err(BarracudaError::InvalidInput {
                message: "Tridiagonal dimensions mismatch".to_string(),
            });
        }

        let mut triplets = Vec::with_capacity(3 * n - 2);

        // Lower diagonal
        for (i, &v) in lower.iter().enumerate() {
            triplets.push((i + 1, i, v));
        }

        // Main diagonal
        for (i, &v) in main.iter().enumerate() {
            triplets.push((i, i, v));
        }

        // Upper diagonal
        for (i, &v) in upper.iter().enumerate() {
            triplets.push((i, i + 1, v));
        }

        Ok(Self::from_triplets(n, n, &triplets))
    }

    /// Number of non-zeros
    #[must_use]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Sparsity (fraction of zeros)
    #[must_use]
    pub fn sparsity(&self) -> f64 {
        let total = self.n_rows * self.n_cols;
        if total == 0 {
            0.0
        } else {
            1.0 - (self.nnz() as f64 / total as f64)
        }
    }

    /// Get value at (row, col) - `O(nnz_row)` lookup
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.n_rows || col >= self.n_cols {
            return 0.0;
        }

        let row_start = self.row_ptr[row];
        let row_end = self.row_ptr[row + 1];

        for i in row_start..row_end {
            if self.col_indices[i] == col {
                return self.values[i];
            }
        }

        0.0
    }

    /// Get diagonal elements
    #[must_use]
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.n_rows.min(self.n_cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Matrix-vector multiplication: y = A * x
    /// # Errors
    /// Returns [`Err`] if vector length does not match matrix column count.
    pub fn matvec(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_cols {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Vector length {} doesn't match matrix columns {}",
                    x.len(),
                    self.n_cols
                ),
            });
        }

        let mut y = vec![0.0; self.n_rows];

        for (y_row, w) in y.iter_mut().zip(self.row_ptr.windows(2)) {
            let row_start = w[0];
            let row_end = w[1];

            for i in row_start..row_end {
                *y_row += self.values[i] * x[self.col_indices[i]];
            }
        }

        Ok(y)
    }

    /// Transpose matrix-vector: y = Aᵀ * x
    /// # Errors
    /// Returns [`Err`] if vector length does not match matrix row count.
    pub fn matvec_transpose(&self, x: &[f64]) -> Result<Vec<f64>> {
        if x.len() != self.n_rows {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Vector length {} doesn't match matrix rows {}",
                    x.len(),
                    self.n_rows
                ),
            });
        }

        let mut y = vec![0.0; self.n_cols];

        for (row, w) in self.row_ptr.windows(2).enumerate() {
            let row_start = w[0];
            let row_end = w[1];
            let x_row = x[row];

            for i in row_start..row_end {
                y[self.col_indices[i]] += self.values[i] * x_row;
            }
        }

        Ok(y)
    }

    /// Scale all values by a constant
    pub fn scale(&mut self, alpha: f64) {
        for v in &mut self.values {
            *v *= alpha;
        }
    }

    /// Convert to dense matrix (column-major)
    #[must_use]
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.n_rows * self.n_cols];

        for row in 0..self.n_rows {
            let row_start = self.row_ptr[row];
            let row_end = self.row_ptr[row + 1];

            for i in row_start..row_end {
                let col = self.col_indices[i];
                dense[row * self.n_cols + col] = self.values[i];
            }
        }

        dense
    }

    /// Check if matrix is symmetric
    #[must_use]
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.n_rows != self.n_cols {
            return false;
        }

        for row in 0..self.n_rows {
            let row_start = self.row_ptr[row];
            let row_end = self.row_ptr[row + 1];

            for i in row_start..row_end {
                let col = self.col_indices[i];
                let val = self.values[i];
                let val_t = self.get(col, row);

                if (val - val_t).abs() > tol {
                    return false;
                }
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_construction() {
        let mut coo = CooMatrix::new(3, 3);
        coo.add(0, 0, 1.0);
        coo.add(0, 2, 2.0);
        coo.add(1, 1, 3.0);
        coo.add(2, 0, 4.0);
        coo.add(2, 2, 5.0);

        assert_eq!(coo.nnz(), 5);
    }

    #[test]
    fn test_csr_from_triplets() {
        let csr = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, 3.0),
                (2, 0, 4.0),
                (2, 2, 5.0),
            ],
        );

        assert_eq!(csr.n_rows, 3);
        assert_eq!(csr.n_cols, 3);
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 5]);
    }

    #[test]
    fn test_csr_get() {
        let csr = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, 3.0),
                (2, 0, 4.0),
                (2, 2, 5.0),
            ],
        );

        assert!((csr.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((csr.get(0, 1) - 0.0).abs() < 1e-10);
        assert!((csr.get(0, 2) - 2.0).abs() < 1e-10);
        assert!((csr.get(1, 1) - 3.0).abs() < 1e-10);
        assert!((csr.get(2, 0) - 4.0).abs() < 1e-10);
        assert!((csr.get(2, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_matvec() {
        // [1 0 2] [1]   [3]   (1*1 + 0*1 + 2*1 = 3)
        // [0 3 0] [1] = [3]   (0*1 + 3*1 + 0*1 = 3)
        // [4 0 5] [1]   [9]   (4*1 + 0*1 + 5*1 = 9)
        let csr = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 2, 2.0),
                (1, 1, 3.0),
                (2, 0, 4.0),
                (2, 2, 5.0),
            ],
        );

        let x = vec![1.0, 1.0, 1.0];
        let y = csr.matvec(&x).unwrap();

        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 3.0).abs() < 1e-10);
        assert!((y[2] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_identity() {
        let eye = CsrMatrix::identity(3);

        assert_eq!(eye.nnz(), 3);
        assert!((eye.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((eye.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((eye.get(2, 2) - 1.0).abs() < 1e-10);
        assert!((eye.get(0, 1) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_diagonal() {
        let mat = CsrMatrix::from_diagonal(&[1.0, 2.0, 3.0]);

        assert_eq!(mat.nnz(), 3);
        assert!((mat.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((mat.get(1, 1) - 2.0).abs() < 1e-10);
        assert!((mat.get(2, 2) - 3.0).abs() < 1e-10);

        let d = mat.diagonal();
        assert_eq!(d, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csr_tridiagonal() {
        let lower = vec![-1.0, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];

        let tri = CsrMatrix::tridiagonal(&lower, &main, &upper).unwrap();

        assert_eq!(tri.n_rows, 3);
        assert_eq!(tri.nnz(), 7);
        assert!((tri.get(0, 0) - 2.0).abs() < 1e-10);
        assert!((tri.get(0, 1) - (-1.0)).abs() < 1e-10);
        assert!((tri.get(1, 0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_csr_sparsity() {
        let csr = CsrMatrix::from_triplets(100, 100, &[(0, 0, 1.0), (50, 50, 2.0)]);

        let sparsity = csr.sparsity();
        assert!(sparsity > 0.99); // 2 non-zeros in 10000 elements
    }

    #[test]
    fn test_csr_symmetric() {
        // Symmetric matrix
        let sym = CsrMatrix::from_triplets(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 1, 2.0),
                (1, 0, 2.0),
                (1, 1, 3.0),
                (1, 2, 4.0),
                (2, 1, 4.0),
                (2, 2, 5.0),
            ],
        );
        assert!(sym.is_symmetric(1e-10));

        // Non-symmetric matrix
        let nonsym = CsrMatrix::from_triplets(3, 3, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0)]);
        assert!(!nonsym.is_symmetric(1e-10));
    }

    #[test]
    fn test_csr_from_triplets_summed() {
        let csr = CsrMatrix::from_triplets_summed(
            3,
            3,
            &[
                (0, 0, 1.0),
                (0, 0, 2.0),
                (1, 1, 3.0),
                (2, 0, 4.0),
                (2, 0, 1.0),
                (2, 2, 5.0),
            ],
        );
        assert!((csr.get(0, 0) - 3.0).abs() < 1e-10);
        assert!((csr.get(1, 1) - 3.0).abs() < 1e-10);
        assert!((csr.get(2, 0) - 5.0).abs() < 1e-10);
        assert!((csr.get(2, 2) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_csr_from_triplets_summed_empty() {
        let csr = CsrMatrix::from_triplets_summed(3, 3, &[]);
        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.n_rows, 3);
    }

    #[test]
    fn test_csr_to_dense() {
        let csr =
            CsrMatrix::from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 3.0), (1, 1, 4.0)]);

        let dense = csr.to_dense();
        assert_eq!(dense, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
