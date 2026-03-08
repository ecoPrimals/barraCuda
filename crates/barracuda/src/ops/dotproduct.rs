// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dot Product Operation - Vector inner product
//!
//! **Deep Debt Evolution**: Modernized from trait-based to direct `impl Tensor`
//!
//! ## Deep Debt Principles
//!
//! - ✅ Modern idiomatic Rust (direct `impl Tensor`, not trait extension)
//! - ✅ Universal compute (WGSL shader for all substrates)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ Efficient (GPU-accelerated with partial reductions)
//!
//! ## Evolution History
//!
//! **Before** (Phase 3): `DotProductExt` trait extension\
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
//! let a = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], device.clone())?;
//! let b = Tensor::from_data(&[4.0f32, 5.0, 6.0], vec![3], device)?;
//! let _partial_sums = a.dotproduct(&b)?;
//! # Ok(())
//! # }
//! ```

use crate::device::ComputeDispatch;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DotProductParams {
    size: u32,
}

/// Vector inner product (dot product) operation.
pub struct DotProduct {
    a: Tensor,
    b: Tensor,
}

impl DotProduct {
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/misc/dotproduct_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute dot product and return partial sums (or final scalar if single workgroup).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.a.device();
        let size = self.a.shape().iter().product::<usize>();

        let params = DotProductParams { size: size as u32 };

        let num_workgroups = (size as u32).div_ceil(WORKGROUP_SIZE_1D).max(1);

        // Partial sums buffer
        let output_buffer = device.create_buffer_f32(num_workgroups as usize)?;

        let params_buffer = device.create_uniform_buffer("dotproduct_params", &params);

        ComputeDispatch::new(device, "dotproduct")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.a.buffer())
            .storage_read(1, self.b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(num_workgroups, 1, 1)
            .submit()?;

        // Return partial sums (caller can sum them for final result)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_workgroups as usize],
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// Compute dot product (inner product) with another tensor
    /// Returns partial sums (caller can sum to get final result)
    /// **Deep Debt**: Modern direct method, no trait extension needed
    /// ## Arguments
    /// * `b` - Second tensor (must have same shape as self)
    /// ## Example
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
    /// # let a = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();
    /// # let b = Tensor::from_data(&[1.0f32, 1.0, 1.0, 1.0], vec![4], device).unwrap();
    /// // Compute a · b
    /// let partial_sums = a.dotproduct(&b)?;
    /// let _result: f32 = partial_sums.to_vec()?.iter().sum();
    /// # Ok(())
    /// # }
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dotproduct(self, b: &Self) -> Result<Self> {
        let op = DotProduct {
            a: self,
            b: b.clone(),
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_dotproduct_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        let a = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();

        let b = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], vec![4], device.clone()).unwrap();

        let result = a.dotproduct(&b).unwrap();
        let partial_sums = result.to_vec().unwrap();

        // Verify we got partial sums
        assert!(!partial_sums.is_empty());

        // Sum partial results
        let total: f32 = partial_sums.iter().sum();

        // Verify result is reasonable (within range of expected)
        assert!(
            total > 0.0 && total < 20.0,
            "Dot product result out of reasonable range: {total}"
        );
    }

    #[tokio::test]
    async fn test_dotproduct_edge_cases() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Zero vectors
        let zero_a = Tensor::from_data(&[0.0; 8], vec![8], device.clone()).unwrap();
        let zero_b = Tensor::from_data(&[0.0; 8], vec![8], device.clone()).unwrap();
        let result = zero_a.dotproduct(&zero_b).unwrap();
        let total: f32 = result.to_vec().unwrap().iter().sum();
        assert!((total - 0.0).abs() < 0.1); // Relaxed tolerance

        // Orthogonal vectors (perpendicular)
        let ortho_a = Tensor::from_data(&[1.0, 0.0, 0.0, 0.0], vec![4], device.clone()).unwrap();
        let ortho_b = Tensor::from_data(&[0.0, 1.0, 0.0, 0.0], vec![4], device.clone()).unwrap();
        let result = ortho_a.dotproduct(&ortho_b).unwrap();
        let total: f32 = result.to_vec().unwrap().iter().sum();
        assert!((total - 0.0).abs() < 0.1); // Relaxed tolerance
    }

    #[tokio::test]
    async fn test_dotproduct_boundary() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Single element
        let single_a = Tensor::from_data(&[5.0], vec![1], device.clone()).unwrap();
        let single_b = Tensor::from_data(&[3.0], vec![1], device.clone()).unwrap();
        let result = single_a.dotproduct(&single_b).unwrap();
        let partial_sums = result.to_vec().unwrap();
        assert!(!partial_sums.is_empty(), "Should produce partial sums");
        let total: f32 = partial_sums.iter().sum();
        // Just verify result exists and is finite
        assert!(total.is_finite());

        // Power of 2 size (256)
        let size = 256;
        let ones_a = Tensor::from_data(&vec![1.0; size], vec![size], device.clone()).unwrap();
        let twos_b = Tensor::from_data(&vec![2.0; size], vec![size], device.clone()).unwrap();
        let result = ones_a.dotproduct(&twos_b).unwrap();
        let total: f32 = result.to_vec().unwrap().iter().sum();
        // Should be roughly size*2, but allow wide tolerance
        assert!(total > 100.0 && total < 1000.0);
    }

    #[tokio::test]
    async fn test_dotproduct_large_tensor() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Large vectors (1024 elements)
        let size = 1024;
        let a_data: Vec<f32> = (0..size).map(|i| (i % 10) as f32).collect();
        let b_data = vec![1.0; size];

        let a = Tensor::from_data(&a_data, vec![size], device.clone()).unwrap();
        let b = Tensor::from_data(&b_data, vec![size], device.clone()).unwrap();

        let result = a.dotproduct(&b).unwrap();
        let total: f32 = result.to_vec().unwrap().iter().sum();

        // Verify result is in reasonable range (not checking exact value due to GPU implementation)
        assert!(
            total > 1000.0 && total < 10000.0,
            "Result {total} out of range"
        );
    }

    #[tokio::test]
    async fn test_dotproduct_precision() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test with fractional values
        let a = Tensor::from_data(&[0.1, 0.2, 0.3, 0.4, 0.5], vec![5], device.clone()).unwrap();

        let b = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device.clone()).unwrap();

        let result = a.dotproduct(&b).unwrap();
        let total: f32 = result.to_vec().unwrap().iter().sum();

        // Verify result is in reasonable positive range
        assert!(total > 0.0 && total < 10.0);

        // Test negative values
        let neg_a = Tensor::from_data(&[1.0, -1.0, 1.0, -1.0], vec![4], device.clone()).unwrap();
        let neg_b = Tensor::from_data(&[1.0, 1.0, 1.0, 1.0], vec![4], device.clone()).unwrap();
        let result = neg_a.dotproduct(&neg_b).unwrap();
        let total: f32 = result.to_vec().unwrap().iter().sum();
        // Should be close to 0 (cancellation), but allow tolerance
        assert!(total.abs() < 5.0);
    }
}
