// SPDX-License-Identifier: AGPL-3.0-or-later
//! Axis-aware tensor operations: `argmax_dim`, `softmax_dim`
//!
//! CPU implementations for Viterbi decoding and similar workloads.
//! No unsafe, no unwrap in non-test code.

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

impl Tensor {
    /// Returns the index of the maximum value along the given axis.
    /// Output shape is the input shape with the specified axis removed.
    /// Returns a tensor of u32 indices; use `to_vec_u32()` to read them.
    /// # Errors
    /// Returns `Err` if axis is out of bounds or tensor is empty.
    pub fn argmax_dim(&self, axis: usize) -> Result<Self> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(BarracudaError::invalid_op(
                "argmax_dim",
                "Empty tensor not supported",
            ));
        }
        if axis >= shape.len() {
            return Err(BarracudaError::invalid_op(
                "argmax_dim",
                format!("Axis {axis} out of bounds for shape {shape:?}"),
            ));
        }

        let data = self.to_vec()?;
        let axis_size = shape[axis];
        let outer_size: usize = shape[0..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();
        let num_reduced = outer_size * inner_size;

        let mut indices = Vec::with_capacity(num_reduced);
        for r in 0..num_reduced {
            let outer = r / inner_size;
            let inner = r % inner_size;
            let base = outer * axis_size * inner_size + inner;
            let stride = inner_size;

            let mut best_idx = 0u32;
            let mut best_val = f32::NEG_INFINITY;
            for j in 0..axis_size {
                let idx = base + j * stride;
                let val = data.get(idx).copied().unwrap_or(f32::NEG_INFINITY);
                if val > best_val {
                    best_val = val;
                    best_idx = j as u32;
                }
            }
            indices.push(best_idx);
        }

        let mut out_shape = shape.to_vec();
        out_shape.remove(axis);
        Tensor::from_data_pod(&indices, out_shape, self.device().clone())
    }

    /// Applies softmax along the specified axis.
    /// Uses numerically stable formulation: subtract max along axis before exp.
    /// Output shape equals input shape. Row-wise when axis=1, column-wise when axis=0.
    /// # Errors
    /// Returns `Err` if axis is out of bounds or tensor is empty.
    pub fn softmax_dim(&self, axis: usize) -> Result<Self> {
        let shape = self.shape();
        if shape.is_empty() {
            return Err(BarracudaError::invalid_op(
                "softmax_dim",
                "Empty tensor not supported",
            ));
        }
        if axis >= shape.len() {
            return Err(BarracudaError::invalid_op(
                "softmax_dim",
                format!("Axis {axis} out of bounds for shape {shape:?}"),
            ));
        }

        let data = self.to_vec()?;
        let axis_size = shape[axis];
        let outer_size: usize = shape[0..axis].iter().product();
        let inner_size: usize = shape[axis + 1..].iter().product();
        let num_slices = outer_size * inner_size;

        let mut result = data.clone();
        for r in 0..num_slices {
            let outer = r / inner_size;
            let inner = r % inner_size;
            let base = outer * axis_size * inner_size + inner;
            let stride = inner_size;

            // Find max along axis (numerical stability)
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..axis_size {
                let idx = base + j * stride;
                if let Some(&v) = data.get(idx) {
                    if v > max_val {
                        max_val = v;
                    }
                }
            }

            // exp(x - max) and sum
            let mut sum = 0.0f32;
            for j in 0..axis_size {
                let idx = base + j * stride;
                let v = data.get(idx).copied().unwrap_or(0.0);
                let exp_val = (v - max_val).exp();
                result[idx] = exp_val;
                sum += exp_val;
            }

            // Normalize
            if sum > 0.0 {
                for j in 0..axis_size {
                    let idx = base + j * stride;
                    result[idx] /= sum;
                }
            }
        }

        Tensor::from_data(&result, shape.to_vec(), self.device().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_argmax_dim_2d_known_positions() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 2x3: row 0 max at index 2, row 1 max at index 0
        let data = vec![1.0, 2.0, 3.0, 5.0, 4.0, 3.0];
        let t = Tensor::from_data(&data, vec![2, 3], device.clone()).unwrap();
        let out = t.argmax_dim(1).unwrap();
        assert_eq!(out.shape(), &[2]);
        let indices = out.to_vec_u32().unwrap();
        assert_eq!(indices, vec![2, 0]);
    }

    #[tokio::test]
    async fn test_argmax_dim_axis0() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 3x2: column 0 max at row 1, column 1 max at row 2
        let data = vec![1.0, 2.0, 5.0, 3.0, 2.0, 6.0];
        let t = Tensor::from_data(&data, vec![3, 2], device.clone()).unwrap();
        let out = t.argmax_dim(0).unwrap();
        assert_eq!(out.shape(), &[2]);
        let indices = out.to_vec_u32().unwrap();
        assert_eq!(indices, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_softmax_dim_axis1_rows_sum_to_one() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_data(&data, vec![2, 3], device.clone()).unwrap();
        let out = t.softmax_dim(1).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        let result = out.to_vec().unwrap();
        assert_eq!(result.len(), 6);
        let row0_sum: f32 = result[0..3].iter().sum();
        let row1_sum: f32 = result[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-5, "row0 sum={row0_sum}");
        assert!((row1_sum - 1.0).abs() < 1e-5, "row1 sum={row1_sum}");
    }

    #[tokio::test]
    async fn test_softmax_dim_axis0_cols_sum_to_one() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let t = Tensor::from_data(&data, vec![3, 2], device.clone()).unwrap();
        let out = t.softmax_dim(0).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        let result = out.to_vec().unwrap();
        let col0_sum: f32 = result[0] + result[2] + result[4];
        let col1_sum: f32 = result[1] + result[3] + result[5];
        assert!((col0_sum - 1.0).abs() < 1e-5, "col0 sum={col0_sum}");
        assert!((col1_sum - 1.0).abs() < 1e-5, "col1 sum={col1_sum}");
    }

    #[tokio::test]
    async fn test_softmax_dim_single_element_axis() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0];
        let t = Tensor::from_data(&data, vec![3, 1], device.clone()).unwrap();
        let out = t.softmax_dim(1).unwrap();
        assert_eq!(out.shape(), &[3, 1]);
        let result = out.to_vec().unwrap();
        assert_eq!(result, vec![1.0, 1.0, 1.0]);
    }

    #[tokio::test]
    async fn test_argmax_dim_single_element_axis() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0];
        let t = Tensor::from_data(&data, vec![3, 1], device.clone()).unwrap();
        let out = t.argmax_dim(1).unwrap();
        assert_eq!(out.shape(), &[3]);
        let indices = out.to_vec_u32().unwrap();
        assert_eq!(indices, vec![0, 0, 0]);
    }

    #[tokio::test]
    async fn test_argmax_dim_axis_out_of_bounds() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let t = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let err = t.argmax_dim(1).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }

    #[tokio::test]
    async fn test_softmax_dim_axis_out_of_bounds() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let t = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let err = t.softmax_dim(1).unwrap_err();
        assert!(err.to_string().contains("out of bounds"));
    }
}
