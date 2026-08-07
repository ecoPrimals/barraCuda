// SPDX-License-Identifier: AGPL-3.0-or-later
//! Min - Reduction operation finding minimum values - Pure WGSL
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

/// Simple min reduction variant (f64 canonical).
pub const WGSL_MIN_SIMPLE: &str = include_str!("../shaders/math/min_simple_f64.wgsl");

/// Basic min reduction shader (f64 canonical).
pub const WGSL_MIN_BASIC: &str = include_str!("../shaders/math/min_f64.wgsl");

/// Min reduction operation
pub struct Min {
    input: Tensor,
    dim: Option<usize>, // None = global min, Some(d) = min along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Min {
    /// Create a new min operation
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
        include_str!("../shaders/math/min_reduce_f64.wgsl")
    }

    /// Get the WGSL shader source for dimension-wise reduction
    fn wgsl_shader_dim() -> &'static str {
        include_str!("../shaders/math/min_dim_f64.wgsl")
    }

    /// Execute the min operation
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, shader compilation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();

        match self.dim {
            None => {
                // Global min reduction
                let size: usize = shape.iter().product();
                // Deep Debt Evolution: Capability-based dispatch
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                // Create output buffer for partial results
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Min Reduce Output"),
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
                let params_buffer = device.create_uniform_buffer("Min Reduce Params", &params);

                ComputeDispatch::new(device, "min_reduce")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups, 1, 1)
                    .submit()?;

                // CPU final reduction of partial workgroup results.
                // A two-pass GPU reduce is worthwhile at >1M elements;
                // below that threshold, readback + CPU reduce is faster.
                let partial_results =
                    device.read_buffer_f32(&output_buffer, num_workgroups as usize)?;
                let global_min = partial_results.iter().fold(f32::INFINITY, |a, &b| a.min(b));

                // Return scalar tensor
                Ok(Tensor::new(vec![global_min], vec![], device.clone()))
            }
            Some(dim) => {
                // Dimension-wise min reduction
                if dim >= shape.len() {
                    return Err(crate::error::BarracudaError::invalid_input(format!(
                        "Dimension {dim} out of range for shape {shape:?}"
                    )));
                }

                let dim_size = shape[dim];
                let outer_size: usize = shape[..dim].iter().product();
                let inner_size: usize = shape[dim + 1..].iter().product();
                let output_size = outer_size * inner_size;

                // Create output buffer
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Min Dim Output"),
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
                let params_buffer = device.create_uniform_buffer("Min Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "min_dim")
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
    /// Find minimum value (global reduction)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, shader compilation fails, GPU
    /// dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn min(&self) -> Result<Self> {
        Min::new(self.clone(), None, false).execute()
    }

    /// Find minimum value along a dimension
    /// # Arguments
    /// * `dim` - Dimension to find min along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, shader compilation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn min_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Min::new(self.clone(), Some(dim), keepdim).execute()
    }

    /// Find minimum value (legacy method for backward compatibility)
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, shader compilation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn min_wgsl(self, dim: Option<usize>) -> Result<Self> {
        match dim {
            None => Min::new(self, None, false).execute(),
            Some(d) => Min::new(self, Some(d), false).execute(),
        }
    }
}
