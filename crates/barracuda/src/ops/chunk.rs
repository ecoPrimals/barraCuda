// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chunk - Split tensor into chunks along dimension
//!
//! **Deep Debt Principles**:
//! - Complete implementation: Uses existing chunk.wgsl shader
//! - Zero hardcoding: All parameters configurable
//! - Self-knowledge: Validates chunk count and dimension
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ChunkParams {
    start_offset: u32, // Start offset in the split dimension for this chunk
    chunk_size: u32,   // Size of this chunk along split dimension
    split_dim: u32,
    dim_size: u32,
    inner_size: u32,
    outer_size: u32,
    output_size: u32,
    _pad1: u32,
}

/// Split tensor into chunks along a dimension.
pub struct Chunk {
    input: Tensor,
    chunks: usize,
    dim: usize,
}

impl Chunk {
    /// Creates a new chunk operation. Splits into `chunks` chunks along `dim`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, chunks: usize, dim: usize) -> Result<Self> {
        if chunks == 0 {
            return Err(BarracudaError::invalid_op(
                "chunk",
                "Cannot split into 0 chunks",
            ));
        }

        let shape = input.shape();
        if dim >= shape.len() {
            return Err(BarracudaError::invalid_op(
                "chunk",
                format!("dim {} exceeds tensor rank {}", dim, shape.len()),
            ));
        }

        // Note: PyTorch allows non-divisible chunks - first (dim_size % chunks) chunks
        // get (dim_size // chunks) + 1 elements, rest get (dim_size // chunks) elements

        Ok(Self { input, chunks, dim })
    }

    /// Executes chunking and returns a vector of chunk tensors.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Vec<Tensor>> {
        let device = self.input.device();
        let shape = self.input.shape();
        let dim_size = shape[self.dim];

        // PyTorch-style chunking: first (dim_size % chunks) chunks get +1 element
        let base_chunk_size = dim_size / self.chunks;
        let extra_chunks = dim_size % self.chunks;

        // Compute sizes
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();

        let mut results = Vec::with_capacity(self.chunks);
        let mut start_offset = 0;

        for chunk_idx in 0..self.chunks {
            // Calculate chunk size: first extra_chunks get +1 element
            let chunk_size = if chunk_idx < extra_chunks {
                base_chunk_size + 1
            } else {
                base_chunk_size
            };

            let output_size = outer_size * chunk_size * inner_size;

            let params = ChunkParams {
                start_offset: start_offset as u32,
                chunk_size: chunk_size as u32,
                split_dim: self.dim as u32,
                dim_size: dim_size as u32,
                inner_size: inner_size as u32,
                outer_size: outer_size as u32,
                output_size: output_size as u32,
                _pad1: 0,
            };

            start_offset += chunk_size;

            let params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Chunk Params"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    });

            let output_buffer = device.create_buffer_f32(output_size)?;

            ComputeDispatch::new(device, "Chunk")
                .shader(include_str!("../shaders/tensor/chunk_f64.wgsl"), "main")
                .uniform(0, &params_buffer)
                .storage_read(1, self.input.buffer())
                .storage_rw(2, &output_buffer)
                .dispatch_1d(output_size as u32)
                .submit()?;

            // Compute output shape
            let mut output_shape = shape.to_vec();
            output_shape[self.dim] = chunk_size;

            results.push(Tensor::from_buffer(
                output_buffer,
                output_shape,
                device.clone(),
            ));
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chunk_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(
            (0..12).map(|i| i as f32).collect(),
            vec![3, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let chunks = Chunk::new(input, 2, 0).unwrap().execute().unwrap();
        assert_eq!(chunks.len(), 2);
        // PyTorch-style: first chunk gets +1 element when not divisible
        // 3 elements into 2 chunks: first gets 2, second gets 1
        assert_eq!(chunks[0].shape(), &[2, 4]);
        assert_eq!(chunks[1].shape(), &[1, 4]);
    }

    #[tokio::test]
    async fn test_chunk_along_dim() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(
            (0..12).map(|i| i as f32).collect(),
            vec![2, 6],
            device.clone(),
        )
        .await
        .unwrap();

        let chunks = Chunk::new(input, 3, 1).unwrap().execute().unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2, 2]);
    }

    #[tokio::test]
    async fn test_chunk_invalid() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        assert!(Chunk::new(input.clone(), 0, 0).is_err());
        // Non-divisible chunks are now allowed (PyTorch-style)
        let chunks = Chunk::new(input, 2, 0).unwrap().execute().unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[1].shape(), &[1]);
    }
}
