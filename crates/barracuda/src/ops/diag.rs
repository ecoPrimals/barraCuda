//! Diagonal matrix operations - Pure WGSL implementation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper
//! - ✅ Two modes: extract diagonal or create diagonal matrix
//!
//! ## Usage
//!
//! ```rust,ignore
//! // Extract diagonal from matrix
//! let matrix = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).await?;
//! let diagonal = matrix.diag_extract()?; // Returns [1.0, 4.0]
//!
//! // Create diagonal matrix from vector
//! let vector = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]).await?;
//! let matrix = vector.diag_create()?; // Returns 3x3 matrix with diagonal [1, 2, 3]
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/diag_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DiagParams {
    size: u32,
    output_size: u32, // elements to process: size for extract, size*size for create
    mode: u32,        // 0 = extract, 1 = create
    _pad: u32,
}

pub struct Diag {
    input: Tensor,
    mode: DiagMode,
}

#[derive(Debug, Clone, Copy)]
pub enum DiagMode {
    Extract, // Matrix → Vector (extract diagonal)
    Create,  // Vector → Matrix (create diagonal matrix)
}

impl Diag {
    pub fn new(input: Tensor, mode: DiagMode) -> Result<Self> {
        let shape = input.shape();

        match mode {
            DiagMode::Extract => {
                // Must be square matrix
                if shape.len() < 2 {
                    return Err(BarracudaError::invalid_op(
                        "diag",
                        "Extract mode requires 2D matrix",
                    ));
                }
                let rows = shape[shape.len() - 2];
                let cols = shape[shape.len() - 1];
                if rows != cols {
                    return Err(BarracudaError::invalid_op(
                        "diag",
                        format!("Extract mode requires square matrix, got {rows}x{cols}"),
                    ));
                }
            }
            DiagMode::Create => {
                // Must be 1D vector
                if shape.len() != 1 {
                    return Err(BarracudaError::invalid_op(
                        "diag",
                        "Create mode requires 1D vector",
                    ));
                }
            }
        }

        Ok(Self { input, mode })
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        let (size, output_size, output_shape) = match self.mode {
            DiagMode::Extract => {
                let n = shape[shape.len() - 1];
                (n, n, vec![n])
            }
            DiagMode::Create => {
                let n = shape[0];
                (n, n * n, vec![n, n])
            }
        };

        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = DiagParams {
            size: size as u32,
            output_size: output_size as u32,
            mode: match self.mode {
                DiagMode::Extract => 0,
                DiagMode::Create => 1,
            },
            _pad: 0,
        };

        let params_buffer = device.create_uniform_buffer("Diag Params", &params);

        let dispatch_size = match self.mode {
            DiagMode::Extract => size,
            DiagMode::Create => output_size,
        };

        ComputeDispatch::new(device, "Diag")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(dispatch_size as u32)
            .submit();

        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_diag_extract() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Matrix: [[1, 2], [3, 4]]
        let matrix = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device)
            .await
            .unwrap();

        let diag = Diag::new(matrix, DiagMode::Extract)
            .unwrap()
            .execute()
            .unwrap();
        let result = diag.to_vec().unwrap();

        assert_eq!(result, vec![1.0, 4.0]);
    }

    #[tokio::test]
    async fn test_diag_create() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let vector = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();

        let matrix = Diag::new(vector, DiagMode::Create)
            .unwrap()
            .execute()
            .unwrap();
        let result = matrix.to_vec().unwrap();

        // Should be: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        let expected = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_diag_extract_3x3() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let matrix = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            device,
        )
        .await
        .unwrap();

        let diag = Diag::new(matrix, DiagMode::Extract)
            .unwrap()
            .execute()
            .unwrap();
        let result = diag.to_vec().unwrap();

        assert_eq!(result, vec![1.0, 5.0, 9.0]);
    }
}
