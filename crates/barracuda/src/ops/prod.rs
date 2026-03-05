// SPDX-License-Identifier: AGPL-3.0-or-later
//! Product reduction - Pure WGSL
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

/// f64 canonical source for simple prod reduction.
/// f64 canonical source for dimension-wise product.
pub const WGSL_PROD_DIM_F64: &str = include_str!("../shaders/reduce/prod_dim_f64.wgsl");

/// f32 derived from f64 canonical source.
static WGSL_PROD_DIM_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(WGSL_PROD_DIM_F64));

/// Product reduction operation
pub struct Prod {
    input: Tensor,
    dim: Option<usize>, // None = global product, Some(d) = product along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Prod {
    /// Create a new product operation
    pub fn new(input: Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            dim,
            keepdim,
        }
    }

    /// Get the WGSL shader source for global reduction
    fn wgsl_shader_reduce() -> &'static str {
        include_str!("../shaders/reduce/prod_reduce.wgsl")
    }

    /// Get the WGSL shader source for dimension-wise reduction
    fn wgsl_shader_dim() -> &'static str {
        &WGSL_PROD_DIM_F32
    }

    /// Execute the product operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();

        match self.dim {
            None => {
                // Global product reduction
                let size: usize = shape.iter().product();
                // Deep Debt Evolution: Capability-based dispatch
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                // Create output buffer for partial results
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Prod Reduce Output"),
                    size: (num_workgroups as usize * std::mem::size_of::<f32>()) as u64,
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
                let params_buffer = device.create_uniform_buffer("Prod Reduce Params", &params);

                ComputeDispatch::new(device, "prod_reduce")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups.max(1), 1, 1)
                    .submit();

                // Read back partial results and reduce them on CPU
                // For now, we'll do a simple CPU reduction of partial results
                // In production, you might want to do a second GPU pass
                let partial_results =
                    device.read_buffer_f32(&output_buffer, num_workgroups as usize)?;
                let global_prod: f32 = partial_results.iter().product();

                // Return scalar tensor
                Ok(Tensor::new(vec![global_prod], vec![], device.clone()))
            }
            Some(dim) => {
                // Dimension-wise product reduction
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
                    label: Some("Prod Dim Output"),
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
                let params_buffer = device.create_uniform_buffer("Prod Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "prod_dim")
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
    /// Compute product (global reduction)
    pub fn prod(&self) -> Result<Self> {
        Prod::new(self.clone(), None, false).execute()
    }

    /// Compute product along a dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to compute product along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    pub fn prod_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Prod::new(self.clone(), Some(dim), keepdim).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn prod_cpu(input: &[f32]) -> f32 {
        input.iter().product()
    }

    #[tokio::test]
    async fn test_prod_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.prod().unwrap().to_vec().unwrap();
        let expected = prod_cpu(&input_data);

        assert!(
            (result[0] - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[tokio::test]
    async fn test_prod_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Contains zero (product = 0)
        let input_data = vec![1.0, 2.0, 0.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.prod().unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-6);

        // All ones (product = 1)
        let input_data = vec![1.0, 1.0, 1.0, 1.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.prod().unwrap().to_vec().unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_prod_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Small values to avoid overflow
        let input_data = vec![1.1, 1.2, 1.3, 1.4, 1.5];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let result = input.prod().unwrap().to_vec().unwrap();
        let expected = prod_cpu(&input_data);

        let rel_error = (result[0] - expected).abs() / expected;
        assert!(rel_error < 1e-3, "Expected {}, got {}", expected, result[0]);
    }

    #[tokio::test]
    async fn test_prod_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 10;
        let input_data: Vec<f32> = (1..=size).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![size], device)
            .await
            .unwrap();
        let result = input.prod().unwrap().to_vec().unwrap();
        let expected = prod_cpu(&input_data);

        let rel_error = (result[0] - expected).abs() / expected;
        assert!(rel_error < 1e-2, "Expected {}, got {}", expected, result[0]);
    }

    #[tokio::test]
    async fn test_prod_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device)
            .await
            .unwrap();
        let gpu_result = input.prod().unwrap().to_vec().unwrap();
        let cpu_result = prod_cpu(&input_data);

        let error = (gpu_result[0] - cpu_result).abs();
        assert!(error < 1e-3, "Error {} exceeds threshold", error);
    }

    #[tokio::test]
    async fn test_prod_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test 2D tensor: [[1, 2, 3], [4, 5, 6]]
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();

        // Product along dim 0 (columns): [4, 10, 18]
        let result = input.prod_dim(0, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0).abs() < 1e-4);
        assert!((result[1] - 10.0).abs() < 1e-4);
        assert!((result[2] - 18.0).abs() < 1e-4);

        // Product along dim 1 (rows): [6, 120]
        let result = input.prod_dim(1, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-4);
        assert!((result[1] - 120.0).abs() < 1e-4);

        // Product along dim 0 with keepdim: [[4, 10, 18]]
        let result = input.prod_dim(0, true).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
    }
}
