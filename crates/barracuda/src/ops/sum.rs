// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sum reduction — GPU-resident, pipeline-cached, buffer-pooled
//!
//! Evolved from per-call buffer allocation to pooled TensorContext path.
//! Dimension-wise results stay GPU-resident; global reductions do a single
//! scalar readback.

use crate::device::pipeline_cache::{BindGroupLayoutSignature, GLOBAL_CACHE};
use crate::device::tensor_context::get_device_context;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Simple sum reduction variant (scalar path). f64 canonical, f32 derived.
/// f64 canonical source for dimension-wise sum.
const WGSL_SUM_DIM_F64: &str = include_str!("../shaders/reduce/sum_dim_f64.wgsl");

/// f32 derived from f64 canonical source.
static WGSL_SUM_DIM_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(WGSL_SUM_DIM_F64));

/// Sum reduction operation
pub struct Sum {
    input: Tensor,
    dim: Option<usize>,
    keepdim: bool,
}

impl Sum {
    /// Create a new sum operation.
    pub fn new(input: Tensor, dim: Option<usize>, keepdim: bool) -> Self {
        Self {
            input,
            dim,
            keepdim,
        }
    }

    fn wgsl_shader_reduce() -> &'static str {
        include_str!("../shaders/reduce/sum_reduce.wgsl")
    }

    fn wgsl_shader_dim() -> &'static str {
        &WGSL_SUM_DIM_F32
    }

    /// Execute the sum operation.
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
                let params_buffer = device.create_uniform_buffer("Sum Reduce Params", &params);

                let layout_sig = BindGroupLayoutSignature::reduction();
                let bind_group = ctx.get_or_create_bind_group(
                    layout_sig,
                    &[input_buffer, &output_buffer, &params_buffer],
                    Some("Sum BG"),
                );

                let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
                    device.device(),
                    adapter_info,
                    Self::wgsl_shader_reduce(),
                    layout_sig,
                    "main",
                    Some("Sum Pipeline"),
                );

                let wg = num_workgroups.max(1);
                ctx.record_operation(move |encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sum Pass"),
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, Some(&*bind_group), &[]);
                    pass.dispatch_workgroups(wg, 1, 1);
                })?;

                let partial_results =
                    device.read_buffer_f32(&output_buffer, num_workgroups as usize)?;
                let global_sum: f32 = partial_results.iter().sum();

                Ok(Tensor::new(vec![global_sum], vec![], device.clone()))
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
                let params_buffer = device.create_uniform_buffer("Sum Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                let layout_sig = BindGroupLayoutSignature::reduction();
                let bind_group = ctx.get_or_create_bind_group(
                    layout_sig,
                    &[input_buffer, &output_buffer, &params_buffer],
                    Some("Sum Dim BG"),
                );

                let pipeline = GLOBAL_CACHE.get_or_create_pipeline(
                    device.device(),
                    adapter_info,
                    Self::wgsl_shader_dim(),
                    layout_sig,
                    "main",
                    Some("Sum Dim Pipeline"),
                );

                let wg = workgroups.max(1);
                ctx.record_operation(move |encoder| {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Sum Dim Pass"),
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
    /// Compute sum (global reduction).
    pub fn sum(&self) -> Result<Self> {
        Sum::new(self.clone(), None, false).execute()
    }

    /// Compute sum along a dimension.
    pub fn sum_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Sum::new(self.clone(), Some(dim), keepdim).execute()
    }

    /// Compute sum (legacy method for backward compatibility).
    pub fn sum_wgsl(self, dim: Option<usize>) -> Result<Self> {
        match dim {
            None => Sum::new(self, None, false).execute(),
            Some(d) => Sum::new(self, Some(d), false).execute(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn sum_cpu(input: &[f32]) -> f32 {
        input.iter().sum()
    }

    #[tokio::test]
    async fn test_sum_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let result = input.sum().unwrap().to_vec().unwrap();
        let expected = sum_cpu(&input_data);

        assert!(
            (result[0] - expected).abs() < 1e-4,
            "Expected {}, got {}",
            expected,
            result[0]
        );
    }

    #[tokio::test]
    async fn test_sum_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![0.0, 0.0, 0.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3], device.clone())
            .await
            .unwrap();
        let result = input.sum().unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-6);

        let input_data = vec![5.0, -3.0, 2.0, -4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.sum().unwrap().to_vec().unwrap();
        let expected = sum_cpu(&input_data);
        assert!((result[0] - expected).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_sum_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1e6, 1e-6, -1e6, 1e-6];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.sum().unwrap().to_vec().unwrap();
        let expected = sum_cpu(&input_data);

        let rel_error = if expected.abs() > 1e-5 {
            (result[0] - expected).abs() / expected.abs()
        } else {
            (result[0] - expected).abs()
        };
        assert!(
            rel_error < 1e-3,
            "Expected {}, got {} (rel error {})",
            expected,
            result[0],
            rel_error
        );
    }

    #[tokio::test]
    async fn test_sum_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let input_data: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![size], device)
            .await
            .unwrap();
        let result = input.sum().unwrap().to_vec().unwrap();
        let expected = sum_cpu(&input_data);

        let rel_error = (result[0] - expected).abs() / expected.abs();
        assert!(rel_error < 1e-3, "Expected {}, got {}", expected, result[0]);
    }

    #[tokio::test]
    async fn test_sum_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.5, 2.5, 3.5, 4.5, 5.5];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let gpu_result = input.sum().unwrap().to_vec().unwrap();
        let cpu_result = sum_cpu(&input_data);

        let error = (gpu_result[0] - cpu_result).abs();
        assert!(error < 1e-4, "Error {} exceeds threshold", error);
    }

    #[tokio::test]
    async fn test_sum_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 3], device.clone())
            .await
            .unwrap();

        let result = input.sum_dim(0, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 5.0).abs() < 1e-4);
        assert!((result[1] - 7.0).abs() < 1e-4);
        assert!((result[2] - 9.0).abs() < 1e-4);

        let result = input.sum_dim(1, false).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-4);
        assert!((result[1] - 15.0).abs() < 1e-4);

        let result = input.sum_dim(0, true).unwrap();
        assert_eq!(result.shape(), &[1, 3]);
    }
}
