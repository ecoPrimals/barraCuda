// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fractional max pool 2D operation - Stochastic pooling with non-integer pooling ratios
//!
//! Improves generalization by introducing randomness
//! Reference: "Fractional Max-Pooling" by Graham (2014)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FractionalMaxPool2dParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    _padding: [u32; 2],
}

/// Fractional max pool 2D operation
pub struct FractionalMaxPool2d {
    input: Tensor,
    pool_seq_h: Tensor,
    pool_seq_w: Tensor,
}

impl FractionalMaxPool2d {
    /// Create fractional max pool 2D operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, pool_seq_h: Tensor, pool_seq_w: Tensor) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!("input must be 4D [B, C, H, W], got shape {shape:?}"),
            ));
        }

        let _ = (shape[2], shape[3]);

        // pool_seq_h should have out_height + 1 elements
        // pool_seq_w should have out_width + 1 elements
        let pool_seq_h_shape = pool_seq_h.shape();
        let pool_seq_w_shape = pool_seq_w.shape();

        if pool_seq_h_shape.len() != 1 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!("pool_seq_h must be 1D, got shape {pool_seq_h_shape:?}"),
            ));
        }

        if pool_seq_w_shape.len() != 1 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!("pool_seq_w must be 1D, got shape {pool_seq_w_shape:?}"),
            ));
        }

        let _ = (pool_seq_h_shape[0] - 1, pool_seq_w_shape[0] - 1);

        if pool_seq_h_shape[0] < 2 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!(
                    "pool_seq_h must have at least 2 elements, got {}",
                    pool_seq_h_shape[0]
                ),
            ));
        }

        if pool_seq_w_shape[0] < 2 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!(
                    "pool_seq_w must have at least 2 elements, got {}",
                    pool_seq_w_shape[0]
                ),
            ));
        }

        Ok(Self {
            input,
            pool_seq_h,
            pool_seq_w,
        })
    }

    /// Execute fractional max pool 2D on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        let pool_seq_h_shape = self.pool_seq_h.shape();
        let pool_seq_w_shape = self.pool_seq_w.shape();
        let out_height = pool_seq_h_shape[0] - 1;
        let out_width = pool_seq_w_shape[0] - 1;
        let output_size = batch_size * channels * out_height * out_width;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = FractionalMaxPool2dParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FractionalMaxPool2d Params"),
            size: std::mem::size_of::<FractionalMaxPool2dParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Convert pool sequences to u32 buffers
        let pool_seq_h_data: Vec<u32> = self
            .pool_seq_h
            .to_vec()?
            .iter()
            .map(|&x| x as u32)
            .collect();
        let pool_seq_w_data: Vec<u32> = self
            .pool_seq_w
            .to_vec()?
            .iter()
            .map(|&x| x as u32)
            .collect();

        let pool_seq_h_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FractionalMaxPool2d PoolSeqH"),
                    contents: bytemuck::cast_slice(&pool_seq_h_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let pool_seq_w_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FractionalMaxPool2d PoolSeqW"),
                    contents: bytemuck::cast_slice(&pool_seq_w_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        ComputeDispatch::new(device, "FractionalMaxPool2d")
            .shader(
                include_str!("../shaders/pooling/fractional_max_pool2d_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_read(1, &pool_seq_h_buffer)
            .storage_read(2, &pool_seq_w_buffer)
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(
                (out_width as u32).div_ceil(8),
                (out_height as u32).div_ceil(8),
                (batch_size * channels) as u32,
            )
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fractional_max_pool2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 2 * 3 * 8 * 8], vec![2, 3, 8, 8], device.clone())
            .await
            .unwrap();

        // Pool sequence: [0, 2, 4, 6, 8] for 4x4 output
        let pool_seq_h =
            Tensor::from_vec_on(vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![5], device.clone())
                .await
                .unwrap();

        let pool_seq_w = Tensor::from_vec_on(vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![5], device)
            .await
            .unwrap();

        let output = FractionalMaxPool2d::new(input, pool_seq_h, pool_seq_w)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Output should be [2, 3, 4, 4]
        assert_eq!(result.len(), 2 * 3 * 4 * 4);
    }
}
