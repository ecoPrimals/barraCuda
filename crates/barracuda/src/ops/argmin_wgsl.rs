// SPDX-License-Identifier: AGPL-3.0-or-later
//! Argmin - Find indices of minimum values - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its dimension
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Argmin operation - Find indices of minimum values
pub struct Argmin {
    input: Tensor,
    dim: Option<usize>, // None = global argmin, Some(d) = argmin along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Argmin {
    /// Create a new argmin operation
    #[must_use]
    pub fn new(input: Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            dim,
            keepdim,
        }
    }

    /// Get the WGSL shader source for global reduction
    fn wgsl_shader_reduce() -> &'static str {
        static SHADER_REDUCE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/math/argmin_reduce_f64.wgsl"
            ).to_string()
        });
        &SHADER_REDUCE
    }

    /// Get the WGSL shader source for dimension-wise reduction
    fn wgsl_shader_dim() -> &'static str {
        static SHADER_DIM: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/math/argmin_f64.wgsl"
            ).to_string()
        });
        &SHADER_DIM
    }

    /// Execute the argmin operation
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();

        match self.dim {
            None => {
                // Global argmin reduction
                let size: usize = shape.iter().product();
                // Deep Debt Evolution: Capability-based dispatch
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                // Create output buffer for partial results (indices)
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Argmin Reduce Output"),
                    size: (num_workgroups as usize * std::mem::size_of::<u32>()) as u64,
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
                let params_buffer = device.create_uniform_buffer("Argmin Reduce Params", &params);

                ComputeDispatch::new(device, "argmin_reduce")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups, 1, 1)
                    .submit()?;

                // Read back partial results and find the global argmin on CPU
                // We need to compare values at the partial indices to find the true global argmin
                let partial_indices =
                    crate::utils::read_buffer_u32(device, &output_buffer, num_workgroups as usize)?;

                // Read the input values at these indices to find the minimum
                let input_data = device.read_buffer_f32(input_buffer, size)?;
                let mut global_min_idx = 0u32;
                let mut global_min_val = f32::INFINITY;

                for &idx in &partial_indices {
                    if (idx as usize) < size {
                        let val = input_data[idx as usize];
                        if val < global_min_val {
                            global_min_val = val;
                            global_min_idx = idx;
                        }
                    }
                }

                // Return scalar tensor (single index)
                Ok(Tensor::new(
                    vec![global_min_idx as f32],
                    vec![],
                    device.clone(),
                ))
            }
            Some(dim) => {
                // Dimension-wise argmin reduction
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
                    label: Some("Argmin Dim Output"),
                    size: (output_size * std::mem::size_of::<u32>()) as u64,
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
                let params_buffer = device.create_uniform_buffer("Argmin Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "argmin_dim")
                    .shader(Self::wgsl_shader_dim(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(workgroups, 1, 1)
                    .submit()?;

                // Read back results
                let output_data =
                    crate::utils::read_buffer_u32(device, &output_buffer, output_size)?;

                // Convert u32 to f32 for tensor
                let output_f32: Vec<f32> = output_data.iter().map(|&x| x as f32).collect();

                // Calculate output shape
                let mut output_shape = shape.to_vec();
                if self.keepdim {
                    output_shape[dim] = 1;
                } else {
                    output_shape.remove(dim);
                }

                Ok(Tensor::new(output_f32, output_shape, device.clone()))
            }
        }
    }
}

impl Tensor {
    /// Find index of minimum value (global reduction)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn argmin(&self) -> Result<Self> {
        Argmin::new(self.clone(), None, false).execute()
    }

    /// Find indices of minimum values along a dimension
    /// # Arguments
    /// * `dim` - Dimension to find min along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn argmin_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Argmin::new(self.clone(), Some(dim), keepdim).execute()
    }

    /// Find indices of minimum values along a dimension (legacy method for backward compatibility)
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn argmin_wgsl(self, dim: usize) -> Result<Self> {
        Argmin::new(self, Some(dim), false).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_argmin_1d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![5.0, 1.0, 3.0, 2.0];
        let input = Tensor::new(data, vec![4], device.clone());

        let output = input.argmin_wgsl(0).unwrap();

        assert_eq!(output.shape(), &[] as &[usize]); // Scalar output
        let result = output.to_vec().unwrap();
        assert_eq!(result[0] as u32, 1); // Index of min value (1.0)
    }

    #[tokio::test]
    async fn test_argmin_2d_dim0() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![4.0, 6.0, 3.0, 2.0, 5.0, 1.0];
        let input = Tensor::new(data, vec![3, 2], device.clone());

        let output = input.argmin_wgsl(0).unwrap();

        assert_eq!(output.shape(), &[2]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0] as u32, 1); // Min in column 0 is at index 1 (value 3.0)
        assert_eq!(result[1] as u32, 2); // Min in column 1 is at index 2 (value 1.0)
    }

    #[tokio::test]
    async fn test_argmin_global() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![5.0, 1.0, 3.0, 2.0];
        let input = Tensor::new(data, vec![4], device.clone());

        let output = input.argmin().unwrap();

        assert_eq!(output.shape(), &[] as &[usize]); // Scalar output
        let result = output.to_vec().unwrap();
        assert_eq!(result[0] as u32, 1); // Index of min value (1.0)
    }

    #[tokio::test]
    async fn test_argmin_dim_keepdim() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![4.0, 6.0, 3.0, 2.0, 5.0, 1.0];
        let input = Tensor::new(data, vec![3, 2], device.clone());

        let output = input.argmin_dim(0, true).unwrap();

        assert_eq!(output.shape(), &[1, 2]); // Dimension kept
        let result = output.to_vec().unwrap();
        assert_eq!(result[0] as u32, 1); // Min in column 0 is at index 1
        assert_eq!(result[1] as u32, 2); // Min in column 1 is at index 2
    }
}
