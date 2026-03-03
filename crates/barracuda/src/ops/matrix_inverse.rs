//! Matrix Inverse - Compute inverse of square matrix
//!
//! **Canonical BarraCuda Pattern**: Struct with new/execute
//!
//! Uses Gauss-Jordan elimination.
//! Note: This is a wrapper around the canonical inverse_wgsl operation.

use crate::error::Result;
use crate::tensor::Tensor;

/// Matrix Inverse operation
pub struct MatrixInverse {
    input: Tensor,
}

impl MatrixInverse {
    /// Create a new matrix inverse operation
    pub fn new(input: Tensor) -> Result<Self> {
        // Validate that input is a square matrix
        let shape = input.shape();
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0], // Will be checked dynamically
                actual: shape.to_vec(),
            });
        }

        Ok(Self { input })
    }

    /// Execute the matrix inverse operation
    /// Delegates to the canonical inverse_wgsl implementation
    pub fn execute(self) -> Result<Tensor> {
        // Use the canonical inverse_wgsl implementation
        use crate::ops::inverse_wgsl::Inverse;
        Inverse::new(self.input).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_matrix_inverse_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // [[4,7],[2,6]]
        let matrix = Tensor::from_vec_on(vec![4.0, 7.0, 2.0, 6.0], vec![2, 2], device)
            .await
            .unwrap();
        let inv = MatrixInverse::new(matrix).unwrap().execute().unwrap();
        assert_eq!(inv.shape(), &[2, 2]);
        let result = inv.to_vec().unwrap();
        assert_eq!(result.len(), 4);
        // Verify: M * M^{-1} ≈ I
        let prod = vec![
            4.0 * result[0] + 7.0 * result[2],
            4.0 * result[1] + 7.0 * result[3],
            2.0 * result[0] + 6.0 * result[2],
            2.0 * result[1] + 6.0 * result[3],
        ];
        assert!((prod[0] - 1.0).abs() < 0.01);
        assert!(prod[1].abs() < 0.01);
        assert!(prod[2].abs() < 0.01);
        assert!((prod[3] - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_matrix_inverse_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Identity matrix
        let matrix = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], device)
            .await
            .unwrap();
        let inv = MatrixInverse::new(matrix).unwrap().execute().unwrap();
        let result = inv.to_vec().unwrap();
        assert!((result[0] - 1.0).abs() < 0.01);
        assert!(result[1].abs() < 0.01);
        assert!(result[2].abs() < 0.01);
        assert!((result[3] - 1.0).abs() < 0.01);
    }
}
