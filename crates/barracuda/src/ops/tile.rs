// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tile - Pure WGSL
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

/// Tile operation (repeat tensor along dimensions)
pub struct Tile {
    input: Tensor,
    repeats: Vec<usize>,
}

impl Tile {
    /// Create a new tile operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, repeats: Vec<usize>) -> Result<Self> {
        let num_dims = input.shape().len();
        if repeats.len() != num_dims {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "Repeats length {} doesn't match tensor rank {}",
                repeats.len(),
                num_dims
            )));
        }

        if repeats.contains(&0) {
            return Err(crate::error::BarracudaError::invalid_input(
                "Repeats must be positive",
            ));
        }

        Ok(Self { input, repeats })
    }

    /// Execute the tile operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let num_dims = input_shape.len();

        // Compute output shape
        let output_shape: Vec<usize> = input_shape
            .iter()
            .zip(self.repeats.iter())
            .map(|(&s, &r)| s * r)
            .collect();
        let total_size: usize = output_shape.iter().product();

        // Compute input strides
        let mut input_strides = vec![1; num_dims];
        for i in (0..num_dims - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Compute output strides
        let mut output_strides = vec![1; num_dims];
        for i in (0..num_dims - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1];
        }

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create buffers for shape and stride data
        let input_shape_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tile Input Shape"),
                    contents: bytemuck::cast_slice(
                        &input_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let output_shape_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tile Output Shape"),
                    contents: bytemuck::cast_slice(
                        &output_shape.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let input_strides_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tile Input Strides"),
                    contents: bytemuck::cast_slice(
                        &input_strides.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let output_strides_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Tile Output Strides"),
                    contents: bytemuck::cast_slice(
                        &output_strides.iter().map(|&x| x as u32).collect::<Vec<_>>(),
                    ),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        // Create output buffer
        let output_buffer = device.create_buffer_f32(total_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            total_size: u32,
            num_dims: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            total_size: total_size as u32,
            num_dims: num_dims as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Tile Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Tile")
            .shader(
                include_str!("../shaders/tensor/tile_f64.wgsl"),
                "main",
            )
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_read(2, &input_shape_buffer)
            .storage_read(3, &output_shape_buffer)
            .storage_read(4, &input_strides_buffer)
            .storage_read(5, &output_strides_buffer)
            .storage_rw(6, &output_buffer)
            .dispatch_1d(total_size as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_tile_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let tiled = Tile::new(input, vec![2]).unwrap().execute().unwrap();
        assert_eq!(tiled.shape(), &vec![6]);
    }

    #[tokio::test]
    async fn test_tile_2d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..6).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3], device).unwrap();

        let tiled = Tile::new(input, vec![2, 1]).unwrap().execute().unwrap();
        assert_eq!(tiled.shape(), &vec![4, 3]);
    }

    #[tokio::test]
    async fn test_tile_invalid_length() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(Tile::new(input, vec![2, 3]).is_err());
    }

    #[tokio::test]
    async fn test_tile_zero_repeat() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        assert!(Tile::new(input, vec![0]).is_err());
    }

    #[tokio::test]
    async fn test_tile_multiple_dims() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 2], device).unwrap();

        let tiled = Tile::new(input, vec![2, 2, 2]).unwrap().execute().unwrap();
        assert_eq!(tiled.shape(), &vec![4, 6, 4]);
    }
}
