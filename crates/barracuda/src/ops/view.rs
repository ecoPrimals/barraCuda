// SPDX-License-Identifier: AGPL-3.0-or-later
//! View operation - Change tensor shape without copying data
//!
//! ## Deep Debt Principles
//!
//! - **Zero-copy**: Memory layout unchanged, only shape metadata changes
//! - **Pure abstraction**: Shader is identity operation, Rust handles reshaping
//! - **Type-safe**: Compile-time shape validation where possible
//!
//! ## Implementation
//!
//! **This is a metadata-only operation by design.**
//!
//! View is fundamentally a metadata operation - the underlying data buffer
//! remains unchanged, only the interpretation of its dimensions changes.
//! The WGSL shader is an identity copy for compatibility, but the real work
//! happens in the Rust wrapper managing tensor metadata.
//!
//! **CPU fallback is acceptable here**: This operation does not perform any
//! computation - it only changes how the tensor's shape is interpreted. The
//! data buffer itself is never modified or copied. This is the correct
//! implementation pattern for view operations.

use crate::error::{BarracudaError, Result};
use crate::shaders::precision;
use crate::tensor::Tensor;

/// WGSL shader source (f64 canonical, downcast to f32 at runtime)
pub static WGSL_VIEW: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    precision::downcast_f64_to_f32_with_transcendentals(include_str!(
        "../shaders/tensor/view_f64.wgsl"
    ))
});

/// View operation parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewParams {
    /// Total number of elements (for validation)
    pub num_elements: u32,
    pub _padding: [u32; 3],
}

/// View operation - change tensor shape without data copy
///
/// View is semantically identical to reshape - it changes the shape metadata
/// without copying or modifying the underlying data buffer. The operation is
/// zero-copy via `Arc<Buffer>` sharing.
///
/// ## Usage
///
/// ```rust,ignore
/// use barracuda::ops::view::View;
///
/// let input = Tensor::zeros([2, 3, 4]).await?;
/// let output = View::new(input, vec![6, 4])?.execute()?;
/// // Output shape: [6, 4], same buffer as input
/// ```
pub struct View {
    input: Tensor,
    new_shape: Vec<usize>,
}

impl View {
    /// Create View operation
    /// # Arguments
    /// * `input` - Input tensor
    /// * `new_shape` - New shape (must have same total number of elements)
    /// # Errors
    /// Returns error if the new shape has a different number of elements than the input.
    pub fn new(input: Tensor, new_shape: Vec<usize>) -> Result<Self> {
        // Validate shapes are compatible
        let old_size: usize = input.shape().iter().product();
        let new_size: usize = new_shape.iter().product();

        if old_size != new_size {
            return Err(BarracudaError::shape_mismatch(
                vec![new_size],
                vec![old_size],
            ));
        }

        Ok(Self { input, new_shape })
    }

    /// Execute view operation on tensor
    ///
    /// This is a metadata-only operation - it returns a new tensor with the
    /// same buffer but different shape metadata. No data is copied.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        // **Zero-Copy Implementation**: wgpu buffers are always contiguous,
        // so view is always safe and zero-copy - we just update metadata!
        //
        // The Arc<Buffer> is cloned (cheap ref count increment), not the buffer.
        // Both tensors share the same GPU memory.
        //
        // This is safe because:
        // 1. Element count is validated in new() (old_size == new_size)
        // 2. wgpu buffers are always contiguous (no striding issues)
        // 3. Arc provides safe shared ownership
        // 4. No unsafe code needed!
        //
        // We use the same pattern as Tensor::reshape - create new Tensor
        // with cloned Arc<Buffer> and new shape metadata
        // Note: We need to access private fields, so we'll use a helper method
        // or construct directly. Since buffer and device are private, we'll
        // use the same approach as reshape - clone the tensor and update shape.
        self.input.reshape(self.new_shape)
    }
}

// Convenience method on Tensor
impl Tensor {
    /// View tensor with new shape (zero-copy operation)
    /// Changes the shape metadata without copying data. The new shape must
    /// have the same total number of elements as the original shape.
    /// # Example
    /// ```rust,ignore
    /// let x = Tensor::zeros([2, 3, 4]).await?;  // [2, 3, 4]
    /// let y = x.view(&[6, 4])?;                 // [6, 4] - same buffer!
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn view(&self, shape: &[usize]) -> Result<Tensor> {
        View::new(self.clone(), shape.to_vec())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_view_2d_to_1d() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let input = Tensor::zeros_on(vec![2, 3], device.clone()).await.unwrap();

        let output = input.view(&[6]).unwrap();

        assert_eq!(output.shape(), &[6]);
        assert_eq!(output.len(), 6);
    }

    #[tokio::test]
    async fn test_view_1d_to_2d() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let input = Tensor::zeros_on(vec![6], device.clone()).await.unwrap();

        let output = input.view(&[2, 3]).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.len(), 6);
    }

    #[tokio::test]
    async fn test_view_3d() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let input = Tensor::zeros_on(vec![2, 3, 4], device.clone())
            .await
            .unwrap();

        let output = input.view(&[6, 4]).unwrap();

        assert_eq!(output.shape(), &[6, 4]);
        assert_eq!(output.len(), 24);
    }

    #[tokio::test]
    async fn test_view_invalid_size() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let input = Tensor::zeros_on(vec![6], device).await.unwrap();

        let result = input.view(&[2, 4]); // Needs 8 elements, but input has 6

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_view_same_shape() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let input = Tensor::zeros_on(vec![2, 3], device.clone()).await.unwrap();

        let output = input.view(&[2, 3]).unwrap();

        assert_eq!(output.shape(), &[2, 3]);
        assert_eq!(output.len(), 6);
    }
}
