// SPDX-License-Identifier: AGPL-3.0-or-later
//! RBF Kernel Evaluation - Radial Basis Function Kernels - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Runtime-configured kernel type
//! - ✅ Fused distance + kernel computation
//!
//! ## Algorithm
//!
//! Computes RBF kernel matrix:
//! ```text
//! Input:  X [N, d], Y [M, d] - point clouds
//! Output: K [N, M] where K[i,j] = φ(‖xᵢ - yⱼ‖)
//!
//! Fused computation: distance + kernel in single pass
//! Avoids N×M intermediate distance matrix
//! ```
//!
//! ## Kernel Functions
//!
//! - **Thin Plate Spline**: φ(r) = r² · log(r) [default, best for physics]
//! - **Gaussian**: φ(r) = exp(-ε²r²)
//! - **Multiquadric**: φ(r) = sqrt(1 + ε²r²)
//! - **Inverse Multiquadric**: φ(r) = 1/sqrt(1 + ε²r²)
//! - **Cubic**: φ(r) = r³
//! - **Quintic**: φ(r) = r⁵
//! - **Linear**: φ(r) = r
//!
//! ## Use Case
//!
//! **RBF Surrogate Learning** (hotSpring physics integration):
//! - Training: K = `rbf_kernel(X_train`, `X_train`) → Cholesky → solve for weights
//! - Prediction: K = `rbf_kernel(X_new`, `X_train`) → matmul with weights
//!
//! ## References
//!
//! - Fasshauer, "Meshfree Approximation Methods with MATLAB"
//! - scipy.interpolate.RBFInterpolator
//! - Used in surrogate-based optimization

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../../shaders/interpolation/rbf_kernel_f64.wgsl");

/// RBF kernel types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
#[derive(Default)]
pub enum RbfKernelType {
    /// Thin Plate Spline: r² · log(r) [default, best for physics]
    #[default]
    ThinPlateSpline = 0,
    /// Gaussian: exp(-ε²r²)
    Gaussian = 1,
    /// Multiquadric: sqrt(1 + ε²r²)
    Multiquadric = 2,
    /// Inverse Multiquadric: 1/sqrt(1 + ε²r²)
    InverseMultiquadric = 3,
    /// Cubic: r³
    Cubic = 4,
    /// Quintic: r⁵
    Quintic = 5,
    /// Linear: r
    Linear = 6,
}

/// RBF kernel evaluation operation
///
/// Computes K[i,j] = φ(‖xᵢ - yⱼ‖) for radial basis function φ
pub struct RbfKernel {
    x: Tensor, // Points X [N, d]
    y: Tensor, // Points Y [M, d]
    kernel_type: RbfKernelType,
    epsilon: f32, // Shape parameter (for Gaussian, MQ, IMQ)
}

impl RbfKernel {
    /// Create new RBF kernel operation
    /// # Arguments
    /// * `x` - First point cloud [N, d]
    /// * `y` - Second point cloud [M, d]
    /// * `kernel_type` - Type of RBF kernel
    /// * `epsilon` - Shape parameter (default 1.0)
    /// # Deep Debt Compliance
    /// - No hardcoded sizes (runtime N, M, d)
    /// - No unsafe blocks
    /// - Agnostic design (works with any point cloud)
    /// - Composable (fuses distance + kernel)
    #[must_use]
    pub fn new(x: Tensor, y: Tensor, kernel_type: RbfKernelType, epsilon: f32) -> Self {
        Self {
            x,
            y,
            kernel_type,
            epsilon,
        }
    }

    /// Execute RBF kernel evaluation on GPU
    /// # Returns
    /// Kernel matrix K [N, M] where K[i,j] = φ(‖xᵢ - yⱼ‖)
    /// # Errors
    /// - Returns error if x and y have different dimensions
    /// - Returns error if x or y are not 2D
    /// # Deep Debt Compliance
    /// - Pure WGSL execution (no CPU fallback)
    /// - Fused distance + kernel (single pass)
    /// - Safe buffer management
    pub fn execute(self) -> Result<Tensor> {
        let device = self.x.device();
        let x_shape = self.x.shape();
        let y_shape = self.y.shape();

        // Validate 2D tensors
        if x_shape.len() != 2 || y_shape.len() != 2 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: x_shape.to_vec(),
            });
        }

        let n_rows = x_shape[0]; // N
        let n_cols = y_shape[0]; // M
        let x_dims = x_shape[1]; // d
        let y_dims = y_shape[1]; // d

        // Validate same dimensionality
        if x_dims != y_dims {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n_rows, x_dims],
                actual: vec![n_cols, y_dims],
            });
        }

        let n_dims = x_dims;

        // Create output buffer for kernel matrix K [N, M]
        let output_size = n_rows * n_cols;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params struct matching WGSL layout
        #[repr(C)]
        #[derive(Copy, Clone, Debug, Pod, Zeroable)]
        struct RbfKernelParams {
            n_rows: u32,
            n_cols: u32,
            n_dims: u32,
            kernel_type: u32,
            epsilon: f32,
            _pad0: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = RbfKernelParams {
            n_rows: n_rows as u32,
            n_cols: n_cols as u32,
            n_dims: n_dims as u32,
            kernel_type: self.kernel_type as u32,
            epsilon: self.epsilon,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device.create_uniform_buffer("RbfKernel Params", &params);

        let workgroup_size = 16;
        let workgroups_x = (n_rows as u32).div_ceil(workgroup_size);
        let workgroups_y = (n_cols as u32).div_ceil(workgroup_size);

        ComputeDispatch::new(device, "RbfKernel")
            .shader(SHADER_F64, "main")
            .storage_read(0, self.x.buffer())
            .storage_read(1, self.y.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x.max(1), workgroups_y.max(1), 1)
            .submit()?;

        let output_size = n_rows * n_cols;
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(
            output_data,
            vec![n_rows, n_cols],
            device.clone(),
        ))
    }
}

/// Tensor extension for RBF kernel evaluation
impl Tensor {
    /// Compute RBF kernel matrix between two point clouds
    /// # Arguments
    /// * `other` - Second point cloud Y [M, d]
    /// * `kernel_type` - Type of RBF kernel
    /// * `epsilon` - Shape parameter (default 1.0)
    /// # Returns
    /// Kernel matrix K [N, M] where K[i,j] = φ(‖xᵢ - yⱼ‖)
    /// # Example
    /// ```ignore
    /// let x = Tensor::from_vec(vec![...], vec![10, 3], device)?;  // 10 points, 3D
    /// let y = Tensor::from_vec(vec![...], vec![20, 3], device)?;  // 20 points, 3D
    /// let k = x.rbf_kernel(&y, RbfKernelType::ThinPlateSpline, 1.0)?;
    /// // k is [10, 20] kernel matrix
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn rbf_kernel(
        &self,
        other: &Self,
        kernel_type: RbfKernelType,
        epsilon: f32,
    ) -> Result<Self> {
        RbfKernel::new(self.clone(), other.clone(), kernel_type, epsilon).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rbf_kernel_same_points() {
        let device = crate::device::test_pool::get_test_device().await;
        // Two identical points should have kernel value = 0 for TPS (r=0)
        let x_data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
        let x = Tensor::from_vec_on(x_data, vec![2, 3], device)
            .await
            .unwrap();

        let k = x
            .rbf_kernel(&x, RbfKernelType::ThinPlateSpline, 1.0)
            .unwrap();
        let kernel = k.to_vec().unwrap();

        // K should be [2, 2]
        assert_eq!(kernel.len(), 4);

        // Diagonal should be 0 for TPS (r=0)
        assert!(
            kernel[0].abs() < 1e-5,
            "K[0,0] should be 0, got {}",
            kernel[0]
        );
        assert!(
            kernel[3].abs() < 1e-5,
            "K[1,1] should be 0, got {}",
            kernel[3]
        );
    }

    #[tokio::test]
    async fn test_rbf_kernel_gaussian() {
        let device = crate::device::test_pool::get_test_device().await;
        // Test Gaussian kernel at origin: should be 1.0
        let x_data = vec![0.0, 0.0, 0.0];
        let x = Tensor::from_vec_on(x_data, vec![1, 3], device)
            .await
            .unwrap();

        let k = x.rbf_kernel(&x, RbfKernelType::Gaussian, 1.0).unwrap();
        let kernel = k.to_vec().unwrap();

        // Gaussian at r=0 should be exp(0) = 1.0
        assert!(
            (kernel[0] - 1.0).abs() < 1e-5,
            "Gaussian at r=0 should be 1.0, got {}",
            kernel[0]
        );
    }

    #[tokio::test]
    async fn test_rbf_kernel_dimensions() {
        let device = crate::device::test_pool::get_test_device().await;
        // Test that output has correct dimensions
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 points, 3D
        let y_data = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]; // 3 points, 3D

        let x = Tensor::from_vec_on(x_data, vec![2, 3], device.clone())
            .await
            .unwrap();
        let y = Tensor::from_vec_on(y_data, vec![3, 3], device)
            .await
            .unwrap();

        let k = x.rbf_kernel(&y, RbfKernelType::Cubic, 1.0).unwrap();
        let shape = k.shape();

        // Should be [2, 3]
        assert_eq!(shape, &[2, 3]);

        let kernel = k.to_vec().unwrap();
        assert_eq!(kernel.len(), 6);
    }
}
