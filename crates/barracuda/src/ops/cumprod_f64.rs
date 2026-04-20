// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cumulative Product (f64) - GPU-accelerated prefix product
//!
//! Computes cumulative product along a dimension in double precision.
//! For input [a, b, c, d], output is [a, a*b, a*b*c, a*b*c*d].
//!
//! **Use cases**:
//! - Probability chains
//! - Running products
//! - Gamma function approximations
//! - Factorial computations
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision via SPIR-V/Vulkan
//! - Zero unsafe code
//! - Self-contained (no external dependencies)

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

use wgpu;

/// Parameters passed to the `cumprod_f64` shader
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct CumprodF64Params {
    size: u32,       // Total number of elements
    dim_size: u32,   // Size of the dimension to scan along
    outer_size: u32, // Product of dimensions before dim
    inner_size: u32, // Product of dimensions after dim
}

/// Cumulative product variants
pub enum CumprodVariant {
    /// Standard inclusive cumprod: [a, a*b, a*b*c, ...]
    Inclusive,
    /// Exclusive cumprod (shifted): [1, a, a*b, ...]
    Exclusive,
    /// Reverse cumprod: [a*b*c*d, b*c*d, c*d, d]
    Reverse,
    /// Log-domain cumprod (numerically stable)
    LogDomain,
}

/// F64 cumulative product operation
pub struct CumprodF64 {
    input: Tensor,
    dim: usize,
    variant: CumprodVariant,
}

impl CumprodF64 {
    /// Create a new inclusive cumprod operation along the specified dimension
    #[must_use]
    pub fn new(input: Tensor, dim: usize) -> Self {
        Self {
            input,
            dim,
            variant: CumprodVariant::Inclusive,
        }
    }

    /// Create an exclusive cumprod (shifted, starts with 1)
    #[must_use]
    pub fn exclusive(input: Tensor, dim: usize) -> Self {
        Self {
            input,
            dim,
            variant: CumprodVariant::Exclusive,
        }
    }

    /// Create a reverse cumprod (from end to start)
    #[must_use]
    pub fn reverse(input: Tensor, dim: usize) -> Self {
        Self {
            input,
            dim,
            variant: CumprodVariant::Reverse,
        }
    }

    /// Create a log-domain cumprod (numerically stable for long sequences)
    #[must_use]
    pub fn log_domain(input: Tensor, dim: usize) -> Self {
        Self {
            input,
            dim,
            variant: CumprodVariant::LogDomain,
        }
    }

    /// WGSL shader source for f64 cumprod
    fn shader() -> &'static str {
        include_str!("../shaders/reduce/cumprod_f64.wgsl")
    }

    fn entry_point(&self) -> &'static str {
        match self.variant {
            CumprodVariant::Inclusive => "cumprod_f64",
            CumprodVariant::Exclusive => "cumprod_exclusive_f64",
            CumprodVariant::Reverse => "cumprod_reverse_f64",
            CumprodVariant::LogDomain => "cumprod_log_f64",
        }
    }

    /// Execute the cumprod operation
    /// # Errors
    /// Returns [`Err`] if dimension is out of range for the tensor rank, buffer allocation fails,
    /// GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let _device = self.input.device();
        let shape = self.input.shape().to_vec();
        let n_dims = shape.len();

        // Validate dimension
        if self.dim >= n_dims {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Dimension {} out of bounds for tensor with {} dimensions",
                    self.dim, n_dims
                ),
            });
        }

        let size: usize = shape.iter().product();
        let dim_size = shape[self.dim];

        // Compute outer_size (product of dims before dim) and inner_size (product after)
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();

        // Handle edge cases
        if outer_size == 0 {
            return self.execute_with_params(size, dim_size, 1, inner_size.max(1));
        }

        self.execute_with_params(size, dim_size, outer_size, inner_size.max(1))
    }

    fn execute_with_params(
        self,
        size: usize,
        dim_size: usize,
        outer_size: usize,
        inner_size: usize,
    ) -> Result<Tensor> {
        let device = self.input.device();

        // Create output buffer
        let output_buffer = device.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("CumprodF64 Output"),
            size: (size * std::mem::size_of::<f64>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params = CumprodF64Params {
            size: size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
        };

        let params_buffer = device.create_uniform_buffer("CumprodF64 Params", &params);

        // Compile shader
        let shader = device.compile_shader_f64(Self::shader(), Some("CumprodF64"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CumprodF64 Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create bind group
        let bind_group = device
            .device()
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CumprodF64 Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.input.buffer().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Create pipeline
        let pipeline_layout =
            device
                .device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("CumprodF64 Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device()
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CumprodF64 Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(self.entry_point()),
                cache: None,
                compilation_options: Default::default(),
            });

        // Dispatch
        let total_pairs = (outer_size * inner_size) as u32;
        let workgroups = total_pairs.div_ceil(WORKGROUP_SIZE_1D);

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("CumprodF64 Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CumprodF64 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(std::iter::once(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }

    /// Execute 1D cumprod directly on a slice (convenience method)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails,
    /// or the device is lost.
    pub async fn execute_1d(device: &Arc<WgpuDevice>, data: &[f64]) -> Result<Vec<f64>> {
        let n = data.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let tensor = Tensor::from_f64_data(data, vec![n], device.clone())?;

        let result = Self::new(tensor, 0).execute()?;

        result.to_f64_vec()
    }

    /// Execute 1D exclusive cumprod directly on a slice
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, buffer readback fails,
    /// or the device is lost.
    pub async fn execute_1d_exclusive(device: &Arc<WgpuDevice>, data: &[f64]) -> Result<Vec<f64>> {
        let n = data.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        let tensor = Tensor::from_f64_data(data, vec![n], device.clone())?;

        let result = Self::exclusive(tensor, 0).execute()?;

        result.to_f64_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_cumprod_f64_1d() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = CumprodF64::execute_1d(&device, &input).await.unwrap();

        // Expected: [1, 2, 6, 24, 120]
        let expected = [1.0, 2.0, 6.0, 24.0, 120.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "Mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[tokio::test]
    async fn test_cumprod_f64_exclusive() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let result = CumprodF64::execute_1d_exclusive(&device, &input)
            .await
            .unwrap();

        // Expected exclusive: [1, 1, 2, 6, 24]
        let expected = [1.0, 1.0, 2.0, 6.0, 24.0];
        for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-10,
                "Mismatch at {i}: got {got}, expected {exp}"
            );
        }
    }

    #[tokio::test]
    async fn test_cumprod_f64_with_identity() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        // All ones should give all ones
        let input = [1.0, 1.0, 1.0, 1.0];
        let result = CumprodF64::execute_1d(&device, &input).await.unwrap();

        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-10,
                "Mismatch at {i}: got {val}, expected 1.0"
            );
        }
    }

    #[tokio::test]
    async fn test_cumprod_f64_precision() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        // Test with small increments to verify f64 precision
        let n = 100;
        let input: Vec<f64> = vec![1.001; n];
        let result = CumprodF64::execute_1d(&device, &input).await.unwrap();

        // Final should be 1.001^100
        let expected_final = 1.001_f64.powi(n as i32);
        let error = (result[n - 1] - expected_final).abs() / expected_final;
        assert!(
            error < 1e-10,
            "Precision error too large: {} (expected {}, got {})",
            error,
            expected_final,
            result[n - 1]
        );
    }
}
