// SPDX-License-Identifier: AGPL-3.0-or-later
//! Slice Assign - In-place slice assignment with strided writes - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its slice parameters
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../shaders/math/slice_assign_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Slice assignment parameters.
#[derive(Debug, Clone)]
pub struct SliceRange {
    /// Start index (inclusive).
    pub start: usize,
    /// End index (exclusive).
    pub end: usize,
    /// Stride between elements.
    pub stride: usize,
}

/// Slice Assign operation - In-place slice assignment
pub struct SliceAssign {
    input: Tensor,
    slice_range: SliceRange,
    values: Tensor,
}

impl SliceAssign {
    /// Create a new slice assign operation
    /// # Errors
    /// Returns [`Err`] if start >= end, end exceeds input size, stride is zero, or values size does not match slice size.
    pub fn new(input: Tensor, slice_range: SliceRange, values: Tensor) -> Result<Self> {
        let input_shape = input.shape();
        let input_size = input_shape.iter().product::<usize>();
        let values_size = values.shape().iter().product::<usize>();

        // Validate slice range
        if slice_range.start >= slice_range.end {
            return Err(BarracudaError::invalid_op(
                "SliceAssign",
                format!(
                    "Start {} must be less than end {}",
                    slice_range.start, slice_range.end
                ),
            ));
        }

        if slice_range.end > input_size {
            return Err(BarracudaError::invalid_op(
                "SliceAssign",
                format!("End {} exceeds input size {}", slice_range.end, input_size),
            ));
        }

        if slice_range.stride == 0 {
            return Err(BarracudaError::invalid_op(
                "SliceAssign",
                "Stride must be greater than zero",
            ));
        }

        // Calculate expected slice size
        let slice_size = (slice_range.end - slice_range.start).div_ceil(slice_range.stride);
        if values_size != slice_size {
            return Err(BarracudaError::invalid_op(
                "SliceAssign",
                format!("Values size {values_size} doesn't match slice size {slice_size}"),
            ));
        }

        Ok(Self {
            input,
            slice_range,
            values,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the slice assign operation (modifies input in-place)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer readback fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size: usize = self.input.shape().iter().product();
        let values_size = self.values.shape().iter().product::<usize>();

        // Access buffers directly (zero-copy)
        let input_buffer = self.input.buffer();
        let values_buffer = self.values.buffer();

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            input_size: u32,
            start: u32,
            end: u32,
            stride: u32,
            values_size: u32,
        }

        let params = Params {
            input_size: input_size as u32,
            start: self.slice_range.start as u32,
            end: self.slice_range.end as u32,
            stride: self.slice_range.stride as u32,
            values_size: values_size as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SliceAssign Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let caps = DeviceCapabilities::from_device(device.as_ref());
        let workgroups = caps.dispatch_1d(values_size as u32);

        ComputeDispatch::new(device.as_ref(), "SliceAssign")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, values_buffer)
            .storage_rw(2, input_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, input_buffer, input_size)?;
        Ok(Tensor::new(
            output_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Assign values to a slice of the tensor (in-place)
    /// # Arguments
    /// * `slice_range` - Slice range (start, end, stride)
    /// * `values` - Values to assign
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch/readback fails (e.g. device lost).
    pub fn slice_assign(self, slice_range: SliceRange, values: Self) -> Result<Self> {
        SliceAssign::new(self, slice_range, values)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_slice_assign_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device.clone());
        let values = Tensor::new(vec![10.0, 20.0], vec![2], device);

        let slice_range = SliceRange {
            start: 1,
            end: 3,
            stride: 1,
        };

        let result = input.slice_assign(slice_range, values).unwrap();
        let output_data = result.to_vec().unwrap();

        // Expected: [1, 10, 20, 4, 5] (indices 1,2 assigned)
        assert_eq!(output_data[0], 1.0);
        assert_eq!(output_data[1], 10.0);
        assert_eq!(output_data[2], 20.0);
        assert_eq!(output_data[3], 4.0);
        assert_eq!(output_data[4], 5.0);
    }

    #[tokio::test]
    async fn test_slice_assign_strided() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device.clone());
        // Stride 2 over [0..5] yields indices 0, 2, 4 → need 3 values
        let values = Tensor::new(vec![10.0, 20.0, 30.0], vec![3], device);

        let slice_range = SliceRange {
            start: 0,
            end: 5,
            stride: 2,
        };

        let result = input.slice_assign(slice_range, values).unwrap();
        let output_data = result.to_vec().unwrap();

        // Expected: [10, 2, 20, 4, 30] (stride 2: indices 0, 2, 4)
        assert_eq!(output_data[0], 10.0);
        assert_eq!(output_data[1], 2.0);
        assert_eq!(output_data[2], 20.0);
        assert_eq!(output_data[3], 4.0);
        assert_eq!(output_data[4], 30.0);
    }
}
