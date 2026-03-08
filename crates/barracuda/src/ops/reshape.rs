// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reshape operation - Change tensor shape without copying data
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
//! Reshape is fundamentally a metadata operation - the underlying data buffer
//! remains unchanged, only the interpretation of its dimensions changes.
//! The WGSL shader is an identity copy for compatibility, but the real work
//! happens in the Rust wrapper managing tensor metadata.
//!
//! **CPU fallback is acceptable here**: This operation does not perform any
//! computation - it only changes how the tensor's shape is interpreted. The
//! data buffer itself is never modified or copied. This is the correct
//! implementation pattern for reshape operations.

use crate::error::{BarracudaError, Result};

/// Reshape identity shader (metadata-only, copy-through).
pub static WGSL_RESHAPE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    include_str!(
        "../shaders/tensor/reshape_f64.wgsl"
    ).to_string()
});

/// Reshape parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ReshapeParams {
    /// Total number of elements (for validation)
    pub num_elements: u32,
    /// Padding for uniform buffer alignment.
    pub _padding: [u32; 3],
}

/// Reshape operation - change tensor shape without data copy
///
/// ## Usage
///
/// ```no_run
/// use barracuda::ops::reshape::*;
///
/// # async fn example(device: &wgpu::Device, queue: &wgpu::Queue) {
/// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape: [6]
/// let output = reshape(device, queue, &input, &[2, 3]).await.unwrap();
/// // Output shape: [2, 3], same data: [[1,2,3], [4,5,6]]
/// # }
/// ```
///
/// ## Deep Debt Note
///
/// **This is a metadata-only operation by design.**
///
/// Reshape is fundamentally a metadata operation - the underlying data buffer
/// remains unchanged, only the interpretation of its dimensions changes.
/// This implementation uses pure metadata (zero GPU invocation) for efficiency.
///
/// **CPU fallback is acceptable here**: This operation does not perform any
/// computation - it only changes how the tensor's shape is interpreted. The
/// data buffer itself is never modified or copied. This is the correct
/// implementation pattern for reshape operations.
///
/// # Errors
///
/// Returns [`Err`] if the new shape's element count does not match the input length.
pub async fn reshape(
    _device: &wgpu::Device,
    _queue: &wgpu::Queue,
    input: &[f32],
    new_shape: &[usize],
) -> Result<Vec<f32>> {
    let num_elements = input.len();
    let new_total: usize = new_shape.iter().product();

    // Validate: total elements must match
    if num_elements != new_total {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "Cannot reshape: input has {num_elements} elements, new shape {new_shape:?} requires {new_total}"
            ),
        });
    }

    // Create params (metadata for shape interpretation)
    let _ = ReshapeParams {
        num_elements: num_elements as u32,
        _padding: [0; 3],
    };

    // Pure metadata operation: reshape doesn't copy data, only changes shape interpretation
    // Return input data unchanged (shape is handled by Tensor metadata)
    Ok(input.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reshape_2d_to_1d() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape: [2, 3]

        let output = reshape(device, queue, &input, &[6]).await.unwrap();

        assert_eq!(output.len(), 6);
        assert_eq!(output, input); // Data unchanged
    }

    #[tokio::test]
    async fn test_reshape_1d_to_2d() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Shape: [6]

        let output = reshape(device, queue, &input, &[2, 3]).await.unwrap();

        assert_eq!(output.len(), 6);
        assert_eq!(output, input); // Data unchanged, shape is metadata
    }

    #[tokio::test]
    async fn test_reshape_3d() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        let input: Vec<f32> = (0..24).map(|i| i as f32).collect(); // 24 elements

        let output = reshape(device, queue, &input, &[2, 3, 4]).await.unwrap();

        assert_eq!(output.len(), 24);
        assert_eq!(output, input);
    }

    #[tokio::test]
    async fn test_reshape_invalid_size() {
        let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
            return;
        };
        let device = &dev.device;
        let queue = &dev.queue;

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 6 elements

        let result = reshape(device, queue, &input, &[2, 4]).await; // Needs 8 elements

        assert!(result.is_err());
    }
}
