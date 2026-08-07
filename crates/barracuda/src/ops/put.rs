// SPDX-License-Identifier: AGPL-3.0-or-later
//! Put - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! NOTE: Uses atomic operations when accumulate=true

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Put operation (scatter with indexing)
pub struct Put {
    output: Tensor,
    indices: Vec<u32>,
    values: Tensor,
    accumulate: bool,
}

impl Put {
    /// Create a new put operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if indices length does not match values size, or any index is out of bounds.
    pub fn new(
        output: Tensor,
        indices: Vec<u32>,
        values: Tensor,
        accumulate: bool,
    ) -> Result<Self> {
        let output_size = output.shape().iter().product::<usize>();
        let values_size = values.shape().iter().product::<usize>();

        if indices.len() != values_size {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "Indices length {} doesn't match values size {}",
                indices.len(),
                values_size
            )));
        }

        // Validate indices are in bounds
        for &idx in &indices {
            if idx as usize >= output_size {
                return Err(crate::error::BarracudaError::invalid_input(format!(
                    "Index {idx} out of bounds for output size {output_size}"
                )));
            }
        }

        Ok(Self {
            output,
            indices,
            values,
            accumulate,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            const S: &str = include_str!("../shaders/tensor/put_f64.wgsl");
            S
        }
    }

    /// Execute the put operation
    /// Note: This modifies the output tensor in-place
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.output.device();
        let output_size: usize = self.output.shape().iter().product();
        let num_values = self.values.shape().iter().product::<usize>();

        // Ensure minimum 32 bytes for WebGPU storage buffer binding requirements
        let byte_size = (output_size * std::mem::size_of::<f32>()).max(32);
        let mut work_contents = self.output.to_vec()?;
        work_contents.resize(byte_size / std::mem::size_of::<f32>(), 0.0);
        let work_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Put Work Buffer"),
                contents: bytemuck::cast_slice(&work_contents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        // Access buffers directly (zero-copy)
        let values_buffer = self.values.buffer();

        // Create indices buffer
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Put Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            output_size: u32,
            num_values: u32,
            accumulate: u32,
            _pad1: u32,
        }

        let params = Params {
            output_size: output_size as u32,
            num_values: num_values as u32,
            accumulate: u32::from(self.accumulate),
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Put Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Put")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, values_buffer)
            .storage_read(2, &indices_buffer)
            .storage_rw(3, &work_buffer)
            .dispatch_1d(num_values as u32)
            .submit()?;

        // Read back via device (ensures GPU writes are visible)
        let output_data = device.read_buffer_f32(&work_buffer, output_size)?;
        Ok(Tensor::new(
            output_data,
            self.output.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_put_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let output = Tensor::from_data(&[0.0, 0.0, 0.0, 0.0], vec![4], device.clone()).unwrap();
        let values = Tensor::from_data(&[10.0, 30.0], vec![2], device).unwrap();

        let result = Put::new(output, vec![0, 2], values, false)
            .unwrap()
            .execute()
            .unwrap();
        let output_data = result.to_vec().unwrap();

        assert_eq!(output_data[0], 10.0);
        assert_eq!(output_data[2], 30.0);
        assert_eq!(output_data[1], 0.0);
        assert_eq!(output_data[3], 0.0);
    }

    #[tokio::test]
    async fn test_put_accumulate() {
        let device = crate::device::test_pool::get_test_device().await;
        let output = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();
        let values = Tensor::from_data(&[10.0, 20.0], vec![2], device).unwrap();

        let result = Put::new(output, vec![0, 1], values, true)
            .unwrap()
            .execute()
            .unwrap();
        let output_data = result.to_vec().unwrap();

        // With accumulate, values are added
        assert_eq!(output_data[0], 11.0);
        assert_eq!(output_data[1], 22.0);
    }

    #[tokio::test]
    async fn test_put_invalid_index() {
        let device = crate::device::test_pool::get_test_device().await;
        let output = Tensor::from_data(&[0.0, 0.0], vec![2], device.clone()).unwrap();
        let values = Tensor::from_data(&[1.0], vec![1], device).unwrap();

        assert!(Put::new(output, vec![5], values, false).is_err());
    }

    #[tokio::test]
    async fn test_put_length_mismatch() {
        let device = crate::device::test_pool::get_test_device().await;
        let output = Tensor::from_data(&[0.0, 0.0], vec![2], device.clone()).unwrap();
        let values = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Put::new(output, vec![0], values, false).is_err());
    }

    #[tokio::test]
    async fn test_put_repeated_indices() {
        let device = crate::device::test_pool::get_test_device().await;
        let output = Tensor::from_data(&[0.0, 0.0], vec![2], device.clone()).unwrap();
        let values = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        // Same index twice - race condition without atomics: either write can win
        let result = Put::new(output, vec![0, 0], values, false)
            .unwrap()
            .execute()
            .unwrap();
        let output_data = result.to_vec().unwrap();
        // GPU non-atomic writes to same location are non-deterministic; accept either
        assert!(output_data[0] == 1.0 || output_data[0] == 2.0);
    }
}
