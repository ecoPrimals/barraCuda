// SPDX-License-Identifier: AGPL-3.0-only
//! L2 Norm reduction - Pure WGSL
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

/// Simple norm reduction variant (scalar path).
#[must_use]
pub fn wgsl_norm_simple() -> &'static str {
    static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
        include_str!("../shaders/misc/norm_simple_f64.wgsl").to_string()
    });
    std::sync::LazyLock::force(&SHADER).as_str()
}

/// Norm reduction operation
pub struct Norm {
    input: Tensor,
    p: f32,             // p-norm parameter (default 2.0 for L2 norm)
    dim: Option<usize>, // None = global norm, Some(d) = norm along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Norm {
    /// Create a new norm operation
    #[must_use]
    pub fn new(input: Tensor, p: f32, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            p,
            dim,
            keepdim,
        }
    }

    /// Get the WGSL shader source for global reduction
    fn wgsl_shader_reduce() -> &'static str {
        include_str!("../shaders/reduce/norm_reduce.wgsl")
    }

    /// Get the WGSL shader source for dimension-wise reduction
    fn wgsl_shader_dim() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/reduce/norm_dim_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the norm operation
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range (for dimension-wise norm), or if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();

        match self.dim {
            None => {
                // Global norm reduction
                let size: usize = shape.iter().product();

                // Deep Debt Evolution: Capability-based dispatch
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                // Create output buffer for partial results
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Norm Reduce Output"),
                    size: (num_workgroups as usize * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                // Create uniform buffer for parameters
                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct Params {
                    size: u32,
                    p: f32,
                }

                let params = Params {
                    size: size as u32,
                    p: self.p,
                };
                let params_buffer = device.create_uniform_buffer("Norm Reduce Params", &params);

                ComputeDispatch::new(device, "norm_reduce")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups, 1, 1)
                    .submit()?;

                // Read back partial results and reduce them on CPU
                let partial_results =
                    device.read_buffer_f32(&output_buffer, num_workgroups as usize)?;
                let sum_power: f32 = partial_results.iter().sum();
                // Compute (sum(|x|^p))^(1/p)
                let global_norm = sum_power.powf(1.0 / self.p);

                // Return scalar tensor
                Ok(Tensor::new(vec![global_norm], vec![], device.clone()))
            }
            Some(dim) => {
                // Dimension-wise norm reduction
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
                    label: Some("Norm Dim Output"),
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
                    p: f32,
                }

                let params = Params {
                    dim_size: dim_size as u32,
                    outer_size: outer_size as u32,
                    inner_size: inner_size as u32,
                    p: self.p,
                };
                let params_buffer = device.create_uniform_buffer("Norm Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "norm_dim")
                    .shader(Self::wgsl_shader_dim(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(workgroups, 1, 1)
                    .submit()?;

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
    /// Compute norm (global reduction, default L2 norm with p=2.0)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn norm(&self) -> Result<Self> {
        Norm::new(self.clone(), 2.0, None, false).execute()
    }

    /// Compute p-norm along a dimension
    /// # Arguments
    /// * `p` - p-norm parameter (e.g., 2.0 for L2 norm, 1.0 for L1 norm)
    /// * `dim` - Dimension to compute norm along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range, or buffer allocation/GPU dispatch/buffer readback fails (e.g. device lost).
    pub fn norm_dim(&self, p: f32, dim: usize, keepdim: bool) -> Result<Self> {
        Norm::new(self.clone(), p, Some(dim), keepdim).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn norm_cpu(input: &[f32], p: f32) -> f32 {
        let sum_power: f32 = input.iter().map(|&x| x.abs().powf(p)).sum();
        sum_power.powf(1.0 / p)
    }

    #[tokio::test]
    async fn test_norm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![3.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2], device)
            .await
            .unwrap();
        let result = input.norm().unwrap().to_vec().unwrap();
        let expected = norm_cpu(&input_data, 2.0);

        assert!(
            (result[0] - expected).abs() < 1e-5,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[tokio::test]
    async fn test_norm_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // All zeros (norm = 0)
        let input_data = vec![0.0, 0.0, 0.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device.clone())
            .await
            .unwrap();
        let result = input.norm().unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-6);

        // Single element
        let input_data = vec![5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![1], device)
            .await
            .unwrap();
        let result = input.norm().unwrap().to_vec().unwrap();
        let expected = norm_cpu(&input_data, 2.0);
        assert!((result[0] - expected).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_norm_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1e5, 1e-5, -1e5, 1e-5];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.norm().unwrap().to_vec().unwrap();
        let expected = norm_cpu(&input_data, 2.0);

        let rel_error = (result[0] - expected).abs() / expected;
        assert!(
            rel_error < 1e-3,
            "Expected {}, got {} (rel error {})",
            expected,
            result[0],
            rel_error
        );
    }

    #[tokio::test]
    async fn test_norm_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 100;
        let input_data: Vec<f32> = (1..=size).map(|i| i as f32).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![size], device)
            .await
            .unwrap();
        let result = input.norm().unwrap().to_vec().unwrap();
        let expected = norm_cpu(&input_data, 2.0);

        let rel_error = (result[0] - expected).abs() / expected;
        assert!(rel_error < 1e-3, "Expected {}, got {}", expected, result[0]);
    }

    #[tokio::test]
    async fn test_norm_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let gpu_result = input.norm().unwrap().to_vec().unwrap();
        let cpu_result = norm_cpu(&input_data, 2.0);

        let error = (gpu_result[0] - cpu_result).abs();
        assert!(error < 1e-4, "Error {error} exceeds threshold");
    }

    #[tokio::test]
    async fn test_norm_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test 2D tensor: [[3, 4], [5, 12]] (3-4-5 and 5-12-13 triangles)
        let input_data = vec![3.0, 4.0, 5.0, 12.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();

        // L2 norm along dim 0 (columns): [sqrt(34), sqrt(160)] ≈ [5.83, 12.65]
        let result = input.norm_dim(2.0, 0, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
        let expected_0 = norm_cpu(&[3.0, 5.0], 2.0);
        let expected_1 = norm_cpu(&[4.0, 12.0], 2.0);
        assert!((result[0] - expected_0).abs() < 1e-4);
        assert!((result[1] - expected_1).abs() < 1e-4);

        // L2 norm along dim 1 (rows): [5.0, 13.0]
        let result = input.norm_dim(2.0, 1, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
        let expected_0 = norm_cpu(&[3.0, 4.0], 2.0);
        let expected_1 = norm_cpu(&[5.0, 12.0], 2.0);
        assert!((result[0] - expected_0).abs() < 1e-4);
        assert!((result[1] - expected_1).abs() < 1e-4);

        // Norm along dim 0 with keepdim: [[sqrt(34), sqrt(160)]]
        let result = input.norm_dim(2.0, 0, true).unwrap();
        assert_eq!(result.shape(), &[1, 2]);
    }
}
