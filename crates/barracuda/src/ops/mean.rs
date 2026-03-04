// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mean reduction - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Workgroup-parallel mean reduction shader (global mean via shared-memory tree).
pub const WGSL_MEAN_REDUCE: &str = include_str!("../shaders/reduce/mean_reduce.wgsl");

/// f64 canonical source for dimension-wise mean.
pub const WGSL_MEAN_DIM_F64: &str = include_str!("../shaders/reduce/mean_dim_f64.wgsl");

/// f32 derived from f64 canonical source.
static WGSL_MEAN_DIM_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(WGSL_MEAN_DIM_F64));

/// f64 mean reduction (tree reduction, partial sums).
pub const WGSL_MEAN_REDUCE_F64: &str = include_str!("../shaders/reduce/mean_reduce_f64.wgsl");

/// Mean reduction operation
pub struct Mean {
    input: Tensor,
    dim: Option<usize>, // None = global mean, Some(d) = mean along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Mean {
    /// Create a new mean operation
    pub fn new(input: Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            dim,
            keepdim,
        }
    }

    fn wgsl_shader_reduce() -> &'static str {
        WGSL_MEAN_REDUCE
    }

    fn wgsl_shader_dim() -> &'static str {
        &WGSL_MEAN_DIM_F32
    }

    /// Execute the mean operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();

        match self.dim {
            None => {
                // Global mean reduction — single-workgroup shader that loops
                // over all elements internally (workgroup_size=1).
                let size: usize = shape.iter().product();

                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Mean Reduce Output"),
                    size: std::mem::size_of::<f32>() as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // Create uniform buffer for parameters
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct Params {
                    size: u32,
                }

                let params = Params { size: size as u32 };
                let params_buffer = device.create_uniform_buffer("Mean Reduce Params", &params);

                ComputeDispatch::new(device, "mean_reduce")
                    .shader(Self::wgsl_shader_reduce(), "mean_reduce")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(1, 1, 1)
                    .submit();

                let result = device.read_buffer_f32(&output_buffer, 1)?;
                Ok(Tensor::new(result, vec![], device.clone()))
            }
            Some(dim) => {
                // Dimension-wise mean reduction
                if dim >= shape.len() {
                    return Err(crate::error::BarracudaError::InvalidInput {
                        message: format!("Dimension {dim} out of range for shape {shape:?}"),
                    });
                }

                let dim_size = shape[dim];
                let outer_size: usize = shape[..dim].iter().product();
                let inner_size: usize = shape[dim + 1..].iter().product();
                let output_size = outer_size * inner_size;

                // Create output buffer
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Mean Dim Output"),
                    size: (output_size * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // Create uniform buffer for parameters
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct Params {
                    dim_size: u32,
                    outer_size: u32,
                    inner_size: u32,
                }

                let params = Params {
                    dim_size: dim_size as u32,
                    outer_size: outer_size as u32,
                    inner_size: inner_size as u32,
                };
                let params_buffer = device.create_uniform_buffer("Mean Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "mean_dim")
                    .shader(Self::wgsl_shader_dim(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(workgroups.max(1), 1, 1)
                    .submit();

                // Read back results
                let output_data = device.read_buffer_f32(&output_buffer, output_size)?;

                // Calculate output shape
                let mut output_shape = shape.to_vec();
                if self.keepdim {
                    output_shape[dim] = 1;
                } else {
                    output_shape.remove(dim);
                }

                Ok(Tensor::new(output_data, output_shape, device.clone()))
            }
        }
    }
}

impl Tensor {
    /// Compute mean (global reduction)
    pub fn mean(&self) -> Result<Self> {
        Mean::new(self.clone(), None, false).execute()
    }

    /// Compute mean along a dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to compute mean along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    pub fn mean_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Mean::new(self.clone(), Some(dim), keepdim).execute()
    }

    /// Compute mean (legacy method for backward compatibility)
    pub fn mean_wgsl(self, dim: Option<usize>) -> Result<Self> {
        match dim {
            None => Mean::new(self, None, false).execute(),
            Some(d) => Mean::new(self, Some(d), false).execute(),
        }
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn mean_cpu(input: &[f32]) -> f32 {
        let sum: f32 = input.iter().sum();
        sum / input.len() as f32
    }

    #[tokio::test]
    async fn test_mean_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let result = input.mean().unwrap().to_vec().unwrap();
        let expected = mean_cpu(&input_data);

        assert!(
            (result[0] - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[tokio::test]
    async fn test_mean_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // All zeros
        let input_data = vec![0.0, 0.0, 0.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device.clone())
            .await
            .unwrap();
        let result = input.mean().unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-6);

        // All same value
        let input_data = vec![5.0, 5.0, 5.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.mean().unwrap().to_vec().unwrap();
        assert!((result[0] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_mean_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1e6, -1e6, 1e-6, -1e-6];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.mean().unwrap().to_vec().unwrap();
        let expected = mean_cpu(&input_data);

        assert!(
            (result[0] - expected).abs() < 1e-3,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[tokio::test]
    async fn test_mean_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let input_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![size], device)
            .await
            .unwrap();
        let result = input.mean().unwrap().to_vec().unwrap();
        let expected = mean_cpu(&input_data);

        let rel_error = (result[0] - expected).abs() / expected.abs();
        assert!(rel_error < 1e-4, "Expected {}, got {}", expected, result[0]);
    }

    #[tokio::test]
    async fn test_mean_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        let input = Tensor::from_vec_on(input_data.clone(), vec![7], device)
            .await
            .unwrap();
        let gpu_result = input.mean().unwrap().to_vec().unwrap();
        let cpu_result = mean_cpu(&input_data);

        let error = (gpu_result[0] - cpu_result).abs();
        assert!(error < 1e-5, "Error {} exceeds threshold", error);
    }

    #[tokio::test]
    async fn test_mean_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test 2D tensor: [[1, 2, 3], [4, 5, 6]]
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();

        // Mean along dim 0 (columns): [2.5, 3.5, 4.5]
        let result = input.mean_dim(0, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.5).abs() < 1e-4);
        assert!((result[1] - 3.5).abs() < 1e-4);
        assert!((result[2] - 4.5).abs() < 1e-4);

        // Mean along dim 1 (rows): [2.0, 5.0]
        let result = input.mean_dim(1, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 2.0).abs() < 1e-4);
        assert!((result[1] - 5.0).abs() < 1e-4);

        // Mean along dim 0 with keepdim: [[2.5, 3.5, 4.5]]
        let result = input.mean_dim(0, true).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
    }
}
