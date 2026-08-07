// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matrix Power - Exponentiation by squaring (GPU implementation)
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Multi-pass iterative (log(n) matmuls)
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates square matrix
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::ops::matmul::MatMul;
use crate::tensor::Tensor;

/// Matrix exponentiation by squaring (A^n for integer n ≥ 0).
pub struct MatrixPower {
    input: Tensor,
    power: i32,
}

impl MatrixPower {
    /// Creates a new matrix power operation. Matrix must be square; power must be non-negative.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, power: i32) -> Result<Self> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "matrix_power",
                "Requires at least 2D tensor",
            ));
        }

        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];

        if rows != cols {
            return Err(BarracudaError::invalid_op(
                "matrix_power",
                format!("Requires square matrix, got {rows}x{cols}"),
            ));
        }

        if power < 0 {
            return Err(BarracudaError::invalid_op(
                "matrix_power",
                "Negative powers not supported (requires matrix inversion)",
            ));
        }

        Ok(Self { input, power })
    }

    fn wgsl_shader() -> &'static str {
        const SHADER: &str = include_str!("../shaders/math/matrix_power_f64.wgsl");
        SHADER
    }

    /// Executes matrix exponentiation and returns A^power.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size = shape[shape.len() - 1];
        let matrix_size = size * size;

        if self.power == 0 {
            // Return identity matrix using WGSL shader
            let identity_buffer = device.create_buffer_f32(matrix_size)?;

            // Create parameters
            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct MatrixPowerParams {
                size: u32,
                _pad1: u32,
                _pad2: u32,
                _pad3: u32,
            }

            let params = MatrixPowerParams {
                size: size as u32,
                _pad1: 0,
                _pad2: 0,
                _pad3: 0,
            };

            let params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("MatrixPower Identity Params"),
                        contents: bytemuck::cast_slice(&[params]),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            ComputeDispatch::new(device, "MatrixPower Identity")
                .shader(Self::wgsl_shader(), "init_identity")
                .uniform(0, &params_buffer)
                .storage_rw(3, &identity_buffer)
                .dispatch(size as u32, size as u32, 1)
                .submit()?;

            let output_shape = shape.to_vec();
            let output_elem_count = output_shape.iter().product::<usize>();
            let output_data =
                crate::utils::read_buffer(device, &identity_buffer, output_elem_count)?;
            return Ok(Tensor::new(output_data, output_shape, device.clone()));
        }

        if self.power == 1 {
            return Ok(self.input);
        }

        // Exponentiation by squaring: M^n
        // result = I, base = M; while n>0: if n odd then result *= base; base *= base; n /= 2
        let mut n = self.power as u32;
        let mut base = self.input.clone();
        let identity_data: Vec<f32> = (0..matrix_size)
            .map(|i| if i % (size + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let mut result = Tensor::from_data(&identity_data, shape.to_vec(), device.clone())?;

        while n > 0 {
            if n % 2 == 1 {
                result = MatMul::new(&result, &base).execute()?;
            }
            base = MatMul::new(&base, &base).execute()?;
            n /= 2;
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_matrix_power_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MatrixPower::new(matrix, 2).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();
        // (2I)^2 = 4I
        assert!((output[0] - 4.0).abs() < 1e-4);
        assert!((output[3] - 4.0).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_matrix_power_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(vec![5.0, 3.0, 2.0, 1.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MatrixPower::new(matrix, 0).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();
        assert!((output[0] - 1.0).abs() < 1e-5);
        assert!(output[1].abs() < 1e-5);
        assert!(output[2].abs() < 1e-5);
        assert!((output[3] - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_matrix_power_one() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MatrixPower::new(matrix.clone(), 1)
            .unwrap()
            .execute()
            .unwrap();
        let output = result.to_vec().unwrap();
        let input = matrix.to_vec().unwrap();
        assert_eq!(output, input);
    }
}
