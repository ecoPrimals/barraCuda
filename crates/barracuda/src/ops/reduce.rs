// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reduce Operation - Aggregation across tensor elements
//!
//! **Deep Debt Evolution**: Modernized from trait-based to direct `impl Tensor`
//!
//! ## Deep Debt Principles
//!
//! - ✅ Modern idiomatic Rust (direct `impl Tensor`, not trait extension)
//! - ✅ Universal compute (WGSL shader for all substrates)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ Agnostic design (operation enum, not hardcoded)
//!
//! ## Evolution History
//!
//! **Before** (Phase 3): `ReduceExt` trait extension\
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::ops::reduce::ReduceOperation;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device()).unwrap();
//! let input = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], device)?;
//! let _sum_tensor = input.reduce(ReduceOperation::Sum)?;
//! # Ok(())
//! # }
//! ```

use crate::device::{ComputeDispatch, DeviceCapabilities};
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceParams {
    size: u32,
    operation: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Reduction operation (sum, max, min, mean) over tensor elements.
pub struct Reduce {
    input: Tensor,
    operation: ReduceOperation,
}

/// Reduction operation for tensor aggregation.
#[derive(Debug, Clone, Copy)]
pub enum ReduceOperation {
    /// Sum of all elements.
    Sum,
    /// Maximum element.
    Max,
    /// Minimum element.
    Min,
    /// Mean (average) of elements.
    Mean,
}

impl ReduceOperation {
    fn to_u32(&self) -> u32 {
        match self {
            Self::Sum => 0,
            Self::Max => 1,
            Self::Min => 2,
            Self::Mean => 3,
        }
    }
}

impl Reduce {
    pub(crate) fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/misc/reduce_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute reduction on GPU; returns partial results (one per workgroup).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.shape().iter().product::<usize>();

        let params = ReduceParams {
            size: size as u32,
            operation: self.operation.to_u32(),
            _pad0: 0,
            _pad1: 0,
        };

        // Dispatch using standard 1D shader workgroup size (256)
        let caps = DeviceCapabilities::from_device(device);
        let num_workgroups = caps.dispatch_1d(size as u32);

        // Initialize output buffer to zeros (COPY_DST for shader write on some backends)
        let output_data = vec![0.0f32; num_workgroups as usize];
        let output_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduce_output"),
                contents: bytemuck::cast_slice(&output_data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reduce_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "reduce")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(num_workgroups, 1, 1)
            .submit()?;

        // Read back partial results (like scatter_wgsl - ensures GPU writes visible)
        let partial_data =
            crate::utils::read_buffer(device, &output_buffer, num_workgroups as usize)?;
        Ok(Tensor::new(
            partial_data,
            vec![num_workgroups as usize],
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// Reduce tensor elements using aggregation operation
    ///
    /// Returns partial reduction results (caller can reduce further if needed)
    ///
    /// **Deep Debt**: Modern direct method, no trait extension needed
    ///
    /// ## Arguments
    ///
    /// * `operation` - Reduce operation (Sum, Max, Min, Mean)
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::ops::reduce::ReduceOperation;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device()).unwrap();
    /// # let input = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], device).unwrap();
    /// // Sum all elements
    /// let partial_sums = input.clone().reduce(ReduceOperation::Sum)?;
    /// let _total: f32 = partial_sums.to_vec()?.iter().sum();
    ///
    /// // Find maximum
    /// let partial_maxes = input.reduce(ReduceOperation::Max)?;
    /// let _max = partial_maxes.to_vec()?.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn reduce(self, operation: ReduceOperation) -> Result<Self> {
        let op = Reduce {
            input: self,
            operation,
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_reduce_sum() {
        let device = crate::device::test_pool::get_test_device().await;

        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device).unwrap();

        let result = input.reduce(ReduceOperation::Sum).unwrap();
        println!("Result shape: {:?}, len: {}", result.shape(), result.len());
        let partial_sums = result.to_vec().unwrap();

        // Sum all partial results
        let total: f32 = partial_sums.iter().sum();
        println!("Partial sums: {partial_sums:?}, Total: {total}");
        assert!((total - 10.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_reduce_max() {
        let device = crate::device::test_pool::get_test_device().await;

        let input = Tensor::from_data(&[1.0, 5.0, 3.0, 2.0], vec![4], device).unwrap();

        let result = input.reduce(ReduceOperation::Max).unwrap();
        let partial_maxes = result.to_vec().unwrap();

        // Max of partial results
        let max_val = partial_maxes
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert!((max_val - 5.0).abs() < 1e-5);
    }
}
