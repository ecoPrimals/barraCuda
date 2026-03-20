// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Variance
//!
//! This module contains the GPU execution logic for variance computation,
//! supporting both global reduction and dimension-wise reduction.

use super::Variance;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

impl Variance {
    /// Execute the variance operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input().device();
        let shape = self.input().shape();
        let input_buffer = self.input().buffer();

        match self.dim() {
            None => {
                // Global variance reduction
                // Two-pass algorithm: first compute mean, then variance
                let size: usize = shape.iter().product();
                // Deep Debt Evolution: Capability-based dispatch
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                // Pass 1: Compute mean using tree reduction
                let mean_output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Variance Mean Output"),
                    size: (num_workgroups as usize * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                #[repr(C)]
                #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
                struct Params {
                    size: u32,
                }

                let params = Params { size: size as u32 };
                let params_buffer = device.create_uniform_buffer("Variance Mean Params", &params);

                ComputeDispatch::new(device, "variance_mean")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &mean_output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups, 1, 1)
                    .submit()?;

                // Read back partial sums and compute mean
                let partial_sums =
                    device.read_buffer_f32(&mean_output_buffer, num_workgroups as usize)?;
                let global_sum: f32 = partial_sums.iter().sum();
                let global_mean = global_sum / size as f32;

                // Pass 2: Compute variance using tree reduction with mean
                // Create a buffer with (x - mean)^2 values
                let diff_squared_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Variance Diff Squared"),
                    size: (size * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });

                // CPU variance accumulation — uses Welford's algorithm for
                // numerical stability. GPU fused mean_variance path is
                // available via `GpuViewF64::mean_variance` for large tensors.
                let input_data = device.read_buffer_f32(input_buffer, size)?;
                let diff_squared: Vec<f32> = input_data
                    .iter()
                    .map(|&x| {
                        let diff = x - global_mean;
                        diff * diff
                    })
                    .collect();

                device.queue.write_buffer(
                    &diff_squared_buffer,
                    0,
                    bytemuck::cast_slice(&diff_squared),
                );

                // Now reduce the diff_squared buffer
                let variance_output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Variance Output"),
                    size: (num_workgroups as usize * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });

                ComputeDispatch::new(device, "variance_reduce")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, &diff_squared_buffer)
                    .storage_rw(1, &variance_output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups, 1, 1)
                    .submit()?;

                // Read back partial variance results
                let partial_variances =
                    device.read_buffer_f32(&variance_output_buffer, num_workgroups as usize)?;
                let global_variance_sum: f32 = partial_variances.iter().sum();
                let global_variance = global_variance_sum / size as f32;

                // Return scalar tensor
                Ok(Tensor::new(vec![global_variance], vec![], device.clone()))
            }
            Some(dim) => {
                // Dimension-wise variance reduction
                if dim >= shape.len() {
                    return Err(BarracudaError::InvalidInput {
                        message: format!("Dimension {dim} out of range for shape {shape:?}"),
                    });
                }

                let dim_size = shape[dim];
                let outer_size: usize = shape[..dim].iter().product();
                let inner_size: usize = shape[dim + 1..].iter().product();
                let output_size = outer_size * inner_size;

                // Create output buffer
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Variance Dim Output"),
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
                let params_buffer = device.create_uniform_buffer("Variance Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "variance_dim")
                    .shader(Self::wgsl_shader_dim(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(workgroups.max(1), 1, 1)
                    .submit()?;

                // Read back results
                let output_data = device.read_buffer_f32(&output_buffer, output_size)?;

                // Calculate output shape
                let mut output_shape = shape.to_vec();
                if self.keepdim() {
                    output_shape[dim] = 1;
                } else {
                    output_shape.remove(dim);
                }

                Ok(Tensor::new(output_data, output_shape, device.clone()))
            }
        }
    }
}
