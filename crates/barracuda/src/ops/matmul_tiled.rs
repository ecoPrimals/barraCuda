// SPDX-License-Identifier: AGPL-3.0-only
//! Tiled Matrix Multiplication - High-performance matmul with shared memory
//!
//! **Deep Debt Evolution**: Modernized from trait-based to direct `impl Tensor`
//!
//! ## Deep Debt Principles
//!
//! - ✅ Modern idiomatic Rust (direct `impl Tensor`, not trait extension)
//! - ✅ Universal compute (WGSL shader for all substrates)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ High performance (tile-based blocking for cache efficiency)
//!
//! ## Evolution History
//!
//! **Before** (Phase 3): `MatmulTiledExt` trait extension\
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
//! let a = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device.clone())?;
//! let b = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], device)?;
//! let _c = a.matmul_tiled(&b)?;  // Result: [2, 2]
//! # Ok(())
//! # }
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MatmulTiledParams {
    m: u32,
    k: u32,
    n: u32,
}

/// Tiled matrix multiplication for cache-efficient GPU matmul.
pub struct MatmulTiled {
    a: Tensor,
    b: Tensor,
}

impl MatmulTiled {
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/math/matmul_tiled_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute tiled matrix multiplication and return the result tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.a.device();
        let a_shape = self.a.shape();
        let b_shape = self.b.shape();

        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(crate::error::BarracudaError::invalid_op(
                "MatmulTiled",
                format!("Expected 2D tensors, got shapes {a_shape:?} and {b_shape:?}"),
            ));
        }

        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        if b_shape[0] != k {
            return Err(crate::error::BarracudaError::invalid_op(
                "MatmulTiled",
                format!("Inner dimensions must match: {} != {}", k, b_shape[0]),
            ));
        }

        let params = MatmulTiledParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
        };

        let output_shape = vec![m, n];
        let output_size = output_shape.iter().product::<usize>() * std::mem::size_of::<f32>();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_tiled_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_uniform_buffer("matmul_tiled_params", &params);

        const TILE_SIZE: u32 = 16;
        ComputeDispatch::new(device, "matmul_tiled")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.a.buffer())
            .storage_read(1, self.b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(
                (n as u32).div_ceil(TILE_SIZE),
                (m as u32).div_ceil(TILE_SIZE),
                1,
            )
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// High-performance tiled matrix multiplication
    ///
    /// Uses tile-based blocking for improved cache locality and performance
    ///
    /// **Deep Debt**: Modern direct method, no trait extension needed
    ///
    /// ## Arguments
    ///
    /// * `b` - Second matrix (columns must match self's rows)
    ///
    /// ## Matrix Dimensions
    ///
    /// - `self`: [M, K]
    /// - `b`: [K, N]
    /// - Result: [M, N]
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
    /// # let a = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device.clone()).unwrap();
    /// # let b = Tensor::from_data(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], device).unwrap();
    /// // C = A × B (optimized with tiling)
    /// let _c = a.matmul_tiled(&b)?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn matmul_tiled(self, b: &Self) -> Result<Self> {
        let op = MatmulTiled {
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
    async fn test_matmul_tiled() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // 2x3 * 3x2 = 2x2
        let a =
            Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], device.clone()).unwrap();

        let b = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], device).unwrap();

        let result = a.matmul_tiled(&b).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(output.len(), 4);
    }
}
