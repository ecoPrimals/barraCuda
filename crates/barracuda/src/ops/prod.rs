// SPDX-License-Identifier: AGPL-3.0-or-later
//! Product reduction — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Evolved from per-call buffer allocation to pooled `TensorContext` path.
//! Dimension-wise results stay GPU-resident; global reductions do a single
//! scalar readback.

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 canonical source for simple prod reduction.
/// f64 canonical source for dimension-wise product.
pub const WGSL_PROD_DIM_F64: &str = include_str!("../shaders/reduce/prod_dim_f64.wgsl");

/// f32 derived from f64 canonical source.
static WGSL_PROD_DIM_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_PROD_DIM_F64.to_string());

/// Product reduction operation
pub struct Prod {
    input: Tensor,
    dim: Option<usize>,
    keepdim: bool,
}

impl Prod {
    /// Create a new product operation.
    #[must_use]
    pub fn new(input: Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            dim,
            keepdim,
        }
    }

    fn wgsl_shader_reduce() -> &'static str {
        include_str!("../shaders/reduce/prod_reduce.wgsl")
    }

    fn wgsl_shader_dim() -> &'static str {
        &WGSL_PROD_DIM_F32
    }

    /// Execute the product operation.
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range (for dimension-wise product), or if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();
        let ctx = get_device_context(device);
        let adapter_info = device.adapter_info();

        match self.dim {
            None => {
                let size: usize = shape.iter().product();
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                let output_buffer = ctx.acquire_pooled_output(num_workgroups as usize);

                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct Params {
                    size: u32,
                }

                let params = Params { size: size as u32 };
                let params_buffer = device.create_uniform_buffer("Prod Reduce Params", &params);

                let layout_sig = BindGroupLayoutSignature::reduction();
                let bind_group = ctx.get_or_create_bind_group(
                    layout_sig,
                    &[input_buffer, &output_buffer, &params_buffer],
                    Some("Prod BG"),
                );

                let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
                    device.device(),
                    adapter_info,
                    Self::wgsl_shader_reduce(),
                    layout_sig,
                    "main",
                    Some("Prod Pipeline"),
                );

                let wg = num_workgroups.max(1);
                ctx.record_operation(move |encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Prod Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&*bind_group), &[]);
                    pass.dispatch_workgroups(wg, 1, 1);
                })?;

                let partial_results =
                    device.read_buffer_f32(&output_buffer, num_workgroups as usize)?;
                let global_prod: f32 = partial_results.iter().product();

                Ok(Tensor::new(vec![global_prod], vec![], device.clone()))
            }
            Some(dim) => {
                if dim >= shape.len() {
                    return Err(crate::error::BarracudaError::InvalidInput {
                        message: format!("Dimension {dim} out of range for shape {shape:?}"),
                    });
                }

                let dim_size = shape[dim];
                let outer_size: usize = shape[..dim].iter().product();
                let inner_size: usize = shape[dim + 1..].iter().product();
                let output_size = outer_size * inner_size;

                let output_buffer = ctx.acquire_pooled_output(output_size);

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

                let layout_sig = BindGroupLayoutSignature::reduction();
                let bind_group = ctx.get_or_create_bind_group(
                    layout_sig,
                    &[input_buffer, &output_buffer, &params_buffer],
                    Some("Prod Dim BG"),
                );

                let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
                    device.device(),
                    adapter_info,
                    Self::wgsl_shader_dim(),
                    layout_sig,
                    "main",
                    Some("Prod Dim Pipeline"),
                );

                let wg = workgroups.max(1);
                ctx.record_operation(move |encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Prod Dim Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&*bind_group), &[]);
                    pass.dispatch_workgroups(wg, 1, 1);
                })?;

                let output_data = device.read_buffer_f32(&output_buffer, output_size)?;

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
    /// Compute product (global reduction).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn prod(&self) -> Result<Self> {
        Prod::new(self.clone(), None, false).execute()
    }

    /// Compute product along a dimension.
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range, or buffer allocation/GPU dispatch/buffer readback fails (e.g. device lost).
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
        let input_data = vec![1.0, 2.0, 0.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.prod().unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-6);

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
        assert!(error < 1e-3, "Error {error} exceeds threshold");
    }

    #[tokio::test]
    async fn test_prod_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();

        let result = input.prod_dim(0, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 4.0).abs() < 1e-4);
        assert!((result[1] - 10.0).abs() < 1e-4);
        assert!((result[2] - 18.0).abs() < 1e-4);

        let result = input.prod_dim(1, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-4);
        assert!((result[1] - 120.0).abs() < 1e-4);

        let result = input.prod_dim(0, true).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
    }
}
