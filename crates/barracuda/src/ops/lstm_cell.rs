// SPDX-License-Identifier: AGPL-3.0-or-later
//! LSTM Cell - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// LSTM Cell operation
pub struct LSTMCell {
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    h_prev: Tensor,
    c_prev: Tensor,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
}

impl LSTMCell {
    /// Create a new LSTM cell operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
    ) -> Result<Self> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_size = input_shape[1..].iter().product::<usize>();

        let hidden_size = h_prev.shape()[1..].iter().product::<usize>();

        // Validate dimensions
        if weight_ih.shape().iter().product::<usize>() != 4 * hidden_size * input_size {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "weight_ih must be [4*hidden_size, input_size]",
            ));
        }

        if weight_hh.shape().iter().product::<usize>() != 4 * hidden_size * hidden_size {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "weight_hh must be [4*hidden_size, hidden_size]",
            ));
        }

        if h_prev.shape()[0] != batch_size
            || h_prev.shape()[1..].iter().product::<usize>() != hidden_size
        {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "h_prev shape mismatch",
            ));
        }

        if c_prev.shape()[0] != batch_size
            || c_prev.shape()[1..].iter().product::<usize>() != hidden_size
        {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "c_prev shape mismatch",
            ));
        }

        Ok(Self {
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h_prev,
            c_prev,
            batch_size,
            input_size,
            hidden_size,
        })
    }

    /// Execute the LSTM cell operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.input.device();

        // Create output buffers for h_next and c_next
        let h_next_size = self.batch_size * self.hidden_size;
        let c_next_size = self.batch_size * self.hidden_size;

        let h_next_buffer = device.create_buffer_f32(h_next_size)?;
        let c_next_buffer = device.create_buffer_f32(c_next_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            input_size: u32,
            hidden_size: u32,
            _padding: u32,
        }

        let params = Params {
            batch_size: self.batch_size as u32,
            input_size: self.input_size as u32,
            hidden_size: self.hidden_size as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LSTMCell Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Combine bias_ih and bias_hh into single buffer [bias_ih..., bias_hh...]
        let bias_ih_data = self.bias_ih.to_vec()?;
        let bias_hh_data = self.bias_hh.to_vec()?;
        let mut bias_combined: Vec<f32> =
            Vec::with_capacity(bias_ih_data.len() + bias_hh_data.len());
        bias_combined.extend_from_slice(&bias_ih_data);
        bias_combined.extend_from_slice(&bias_hh_data);

        let bias_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LSTMCell Bias Combined"),
                contents: bytemuck::cast_slice(&bias_combined),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "LSTMCell")
            .shader(
                include_str!("../shaders/rnn/lstm_cell_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.weight_ih.buffer())
            .storage_read(2, self.weight_hh.buffer())
            .storage_read(3, &bias_buffer)
            .storage_read(4, self.h_prev.buffer())
            .storage_read(5, self.c_prev.buffer())
            .storage_rw(6, &h_next_buffer)
            .storage_rw(7, &c_next_buffer)
            .uniform(8, &params_buffer)
            .dispatch_1d(self.batch_size as u32)
            .submit()?;

        let h_size = self.batch_size * self.hidden_size;
        let h_data = crate::utils::read_buffer(device, &h_next_buffer, h_size)?;
        let c_data = crate::utils::read_buffer(device, &c_next_buffer, h_size)?;

        let h_next = Tensor::new(
            h_data,
            vec![self.batch_size, self.hidden_size],
            device.clone(),
        );
        let c_next = Tensor::new(
            c_data,
            vec![self.batch_size, self.hidden_size],
            device.clone(),
        );

        Ok((h_next, c_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lstm_cell_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 2;
        let input_size = 4;
        let hidden_size = 8;

        let input = Tensor::from_vec_on(
            vec![0.5; batch_size * input_size],
            vec![batch_size, input_size],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_ih = Tensor::from_vec_on(
            vec![0.01; 4 * hidden_size * input_size],
            vec![4 * hidden_size, input_size],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_hh = Tensor::from_vec_on(
            vec![0.01; 4 * hidden_size * hidden_size],
            vec![4 * hidden_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias_ih = Tensor::from_vec_on(
            vec![0.0; 4 * hidden_size],
            vec![4 * hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias_hh = Tensor::from_vec_on(
            vec![0.0; 4 * hidden_size],
            vec![4 * hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let h_prev = Tensor::from_vec_on(
            vec![0.0; batch_size * hidden_size],
            vec![batch_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let c_prev = Tensor::from_vec_on(
            vec![0.0; batch_size * hidden_size],
            vec![batch_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let (h_next, c_next) = LSTMCell::new(
            input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev, c_prev,
        )
        .unwrap()
        .execute()
        .unwrap();

        assert_eq!(h_next.shape(), &[batch_size, hidden_size]);
        assert_eq!(c_next.shape(), &[batch_size, hidden_size]);
    }
}
