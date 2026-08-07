// SPDX-License-Identifier: AGPL-3.0-or-later
//! Take - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Take operation (advanced indexing/gather)
pub struct Take {
    input: Tensor,
    indices: Vec<u32>,
}

impl Take {
    /// Create a new take operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, indices: Vec<u32>) -> Result<Self> {
        let input_size = input.shape().iter().product::<usize>();
        if indices.iter().any(|&idx| idx as usize >= input_size) {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "Index out of bounds: input_size={input_size}, indices={indices:?}"
            )));
        }
        Ok(Self { input, indices })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        const S: &str = include_str!("../shaders/tensor/take_f64.wgsl");
        S
    }

    /// Execute the take operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let output_size = self.indices.len();

        if output_size == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create indices buffer
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Take Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer (ensure minimum 32 bytes for WebGPU storage binding)
        let output_byte_size = (output_size * std::mem::size_of::<f32>()).max(32);
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Take Output Buffer"),
            size: output_byte_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            output_size: u32,
            input_size: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let input_size = self.input.shape().iter().product::<usize>();
        let params = Params {
            output_size: output_size as u32,
            input_size: input_size as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Take Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Take")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_read(2, &indices_buffer)
            .storage_rw(3, &output_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(output_data, vec![output_size], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_take_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[10.0, 20.0, 30.0, 40.0], vec![4], device).unwrap();

        let result = Take::new(input, vec![0, 2, 1]).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 3);
        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 30.0);
        assert_eq!(output[2], 20.0);
    }

    #[tokio::test]
    async fn test_take_repeated() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let result = Take::new(input, vec![0, 0, 1, 1, 2])
            .unwrap()
            .execute()
            .unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 5);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 2.0);
        assert_eq!(output[4], 3.0);
    }

    #[tokio::test]
    async fn test_take_large() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1000], device).unwrap();

        let indices: Vec<u32> = (0..100).map(|i| (i * 10) as u32).collect();
        let result = Take::new(input, indices).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 100);
        for (i, &val) in output.iter().enumerate() {
            assert_eq!(val, (i * 10) as f32);
        }
    }

    #[tokio::test]
    async fn test_take_empty() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let result = Take::new(input, vec![]).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 0);
    }

    #[tokio::test]
    async fn test_take_invalid() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Take::new(input, vec![0, 5]).is_err());
    }
}
