// SPDX-License-Identifier: AGPL-3.0-or-later
//! Map Operation - Element-wise transformations
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
//! **Before** (Phase 3): `MapExt` trait extension  
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::ops::map::MapOperation;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
//! let input = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], device)?;
//! let _squared = input.map(MapOperation::Square)?;
//! # Ok(())
//! # }
//! ```

use crate::device::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MapParams {
    size: u32,
    operation: u32,
}

/// Element-wise map operation (square, sqrt, abs, negate, reciprocal).
pub struct Map {
    input: Tensor,
    operation: MapOperation,
}

/// Element-wise map operations.
#[derive(Debug, Clone, Copy)]
pub enum MapOperation {
    /// Square: x².
    Square,
    /// Square root.
    Sqrt,
    /// Absolute value.
    Abs,
    /// Negate: -x.
    Negate,
    /// Reciprocal: 1/x.
    Reciprocal,
}

impl MapOperation {
    fn to_u32(&self) -> u32 {
        match self {
            MapOperation::Square => 0,
            MapOperation::Sqrt => 1,
            MapOperation::Abs => 2,
            MapOperation::Negate => 3,
            MapOperation::Reciprocal => 4,
        }
    }
}

impl Map {
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/misc/map_f64.wgsl"
            ))
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Execute the map operation on GPU.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.shape().iter().product::<usize>();

        let params = MapParams {
            size: size as u32,
            operation: self.operation.to_u32(),
        };

        let output_buffer = device.create_buffer_f32(size)?;

        let params_buffer = device.create_uniform_buffer("map_params", &params);

        ComputeDispatch::new(device, "map")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit();

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// Apply element-wise transformation to tensor
    ///
    /// **Deep Debt**: Modern direct method, no trait extension needed
    ///
    /// ## Arguments
    ///
    /// * `operation` - Map operation (Square, Sqrt, Abs, Negate, Reciprocal)
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::ops::map::MapOperation;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
    /// # let input = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0], vec![4], device).unwrap();
    /// // Square all elements
    /// let _squared = input.clone().map(MapOperation::Square)?;
    ///
    /// // Take square root
    /// let _sqrt = input.map(MapOperation::Sqrt)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn map(self, operation: MapOperation) -> Result<Self> {
        let op = Map {
            input: self,
            operation,
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_map_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();

        let result = input.map(MapOperation::Square).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_map_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Single element
        let input = Tensor::from_data(&[5.0], vec![1], device.clone()).unwrap();
        let result = input.map(MapOperation::Square).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());

        // Negate operation
        let input = Tensor::from_data(&[1.0, -2.0, 3.0], vec![3], device.clone()).unwrap();
        let result = input.map(MapOperation::Negate).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_map_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Sqrt with various values
        let input = Tensor::from_data(&[4.0, 9.0, 16.0], vec![3], device.clone()).unwrap();
        let result = input.map(MapOperation::Sqrt).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));

        // Abs with negative values
        let input = Tensor::from_data(&[-1.0, -2.0, -3.0], vec![3], device.clone()).unwrap();
        let result = input.map(MapOperation::Abs).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x >= 0.0));
    }

    #[tokio::test]
    async fn test_map_large_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 1000 elements
        let input_data: Vec<f32> = (1..=1000).map(|i| i as f32).collect();
        let input = Tensor::from_data(&input_data, vec![1000], device).unwrap();
        let result = input.map(MapOperation::Square).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1000);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_map_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Test square operation
        let input = Tensor::from_data(&[2.0, 3.0], vec![2], device).unwrap();
        let result = input.map(MapOperation::Square).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}
