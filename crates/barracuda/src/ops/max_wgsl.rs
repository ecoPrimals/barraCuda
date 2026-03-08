// SPDX-License-Identifier: AGPL-3.0-or-later
//! Max - Reduction operation finding maximum values - Pure WGSL
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

/// Simple max reduction variant (f64 canonical).
const WGSL_MAX_SIMPLE_F64: &str = include_str!("../shaders/math/max_simple_f64.wgsl");

/// Simple max reduction variant (f32 derived from f64).
pub static WGSL_MAX_SIMPLE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    WGSL_MAX_SIMPLE_F64.to_string()
});

/// f64 is the canonical source — math is universal, precision is silicon.
const WGSL_MAX_BASIC_F64: &str = include_str!("../shaders/math/max_f64.wgsl");

/// Basic max reduction shader (f32 derived from f64).
pub static WGSL_MAX_BASIC: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_MAX_BASIC_F64.to_string());

/// Max reduction operation
pub struct Max {
    input: Tensor,
    dim: Option<usize>, // None = global max, Some(d) = max along dimension d
    keepdim: bool,      // Whether to keep dimension with size 1
}

impl Max {
    /// Create a new max operation
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
                "../shaders/math/max_reduce_f64.wgsl"
            ).to_string()
        });
        &SHADER_REDUCE
    }

    /// Get the WGSL shader source for dimension-wise reduction
    fn wgsl_shader_dim() -> &'static str {
        static SHADER_DIM: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/math/max_dim_f64.wgsl"
            ).to_string()
        });
        &SHADER_DIM
    }

    /// Execute the max operation
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let input_buffer = self.input.buffer();

        match self.dim {
            None => {
                // Global max reduction
                let size: usize = shape.iter().product();
                // Deep Debt Evolution: Capability-based dispatch
                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let num_workgroups = (size as u32).div_ceil(optimal_wg_size);

                // Create output buffer for partial results
                let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Max Reduce Output"),
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
                let params_buffer = device.create_uniform_buffer("Max Reduce Params", &params);

                ComputeDispatch::new(device, "max_reduce")
                    .shader(Self::wgsl_shader_reduce(), "main")
                    .storage_read(0, input_buffer)
                    .storage_rw(1, &output_buffer)
                    .uniform(2, &params_buffer)
                    .dispatch(num_workgroups, 1, 1)
                    .submit()?;

                // Read back partial results and reduce them on CPU
                // For now, we'll do a simple CPU reduction of partial results
                // In production, you might want to do a second GPU pass
                let partial_results =
                    device.read_buffer_f32(&output_buffer, num_workgroups as usize)?;
                let global_max = partial_results
                    .iter()
                    .fold(f32::NEG_INFINITY, |a: f32, &b| a.max(b));

                // Return scalar tensor
                Ok(Tensor::new(vec![global_max], vec![], device.clone()))
            }
            Some(dim) => {
                // Dimension-wise max reduction
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
                    label: Some("Max Dim Output"),
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
                let params_buffer = device.create_uniform_buffer("Max Dim Params", &params);

                let caps = DeviceCapabilities::from_device(device);
                let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups = (output_size as u32).div_ceil(optimal_wg_size);

                ComputeDispatch::new(device, "max_dim")
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
    /// Find maximum value (global reduction)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or buffer
    /// readback fails (e.g. device lost).
    pub fn max(&self) -> Result<Self> {
        Max::new(self.clone(), None, false).execute()
    }

    /// Find maximum value along a dimension
    /// # Arguments
    /// * `dim` - Dimension to find max along
    /// * `keepdim` - Whether to keep the reduced dimension with size 1
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn max_dim(&self, dim: usize, keepdim: bool) -> Result<Self> {
        Max::new(self.clone(), Some(dim), keepdim).execute()
    }

    /// Find maximum value (legacy method for backward compatibility)
    /// # Errors
    /// Returns [`Err`] if `dim` is out of range for the tensor shape, buffer
    /// allocation fails, GPU dispatch fails, or buffer readback fails (e.g. device lost).
    pub fn max_wgsl(self, dim: Option<usize>) -> Result<Self> {
        match dim {
            None => Max::new(self, None, false).execute(),
            Some(d) => Max::new(self, Some(d), false).execute(),
        }
    }
}
