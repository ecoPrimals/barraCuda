// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bidirectional LSTM operation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Processes sequence in both forward and backward directions

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BiLSTMParams {
    batch_size: u32,
    seq_len: u32,
    input_size: u32,
    hidden_size: u32,
    direction: u32, // 0 = forward, 1 = backward
}

/// Bidirectional LSTM: processes sequence in forward and backward directions.
pub struct BiLSTM {
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    direction: u32,
}

impl BiLSTM {
    /// Create `BiLSTM` operation
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
        direction: u32,
    ) -> Result<Self> {
        // Validate direction
        if direction > 1 {
            return Err(BarracudaError::invalid_op(
                "BiLSTM",
                format!("direction must be 0 (forward) or 1 (backward), got {direction}"),
            ));
        }

        Ok(Self {
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            direction,
        })
    }

    /// Execute `BiLSTM` on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();

        if input_shape.len() != 3 {
            return Err(BarracudaError::invalid_op(
                "BiLSTM",
                format!("input must be 3D [seq_len, batch, input_size], got shape {input_shape:?}"),
            ));
        }

        let seq_len = input_shape[0];
        let batch_size = input_shape[1];
        let input_size = input_shape[2];
        let hidden_size = self.bias_ih.len() / 4; // 4 gates: i, f, g, o

        // Create output buffer: [seq_len, batch, hidden_size]
        let output_size = seq_len * batch_size * hidden_size;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create h_state and c_state buffers: [batch, hidden_size]
        let state_size = batch_size * hidden_size;
        let h_state_buffer = device.create_buffer_f32(state_size)?;
        let c_state_buffer = device.create_buffer_f32(state_size)?;

        // Initialize states to zero
        device.queue.write_buffer(
            &h_state_buffer,
            0,
            bytemuck::cast_slice(&vec![0.0f32; state_size]),
        );
        device.queue.write_buffer(
            &c_state_buffer,
            0,
            bytemuck::cast_slice(&vec![0.0f32; state_size]),
        );

        let params = BiLSTMParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            input_size: input_size as u32,
            hidden_size: hidden_size as u32,
            direction: self.direction,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BiLSTM Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "BiLSTM")
            .shader(include_str!("../shaders/rnn/bi_lstm_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.weight_ih.buffer())
            .storage_read(2, self.weight_hh.buffer())
            .storage_read(3, self.bias_ih.buffer())
            .storage_read(4, self.bias_hh.buffer())
            .storage_rw(5, &h_state_buffer)
            .storage_rw(6, &c_state_buffer)
            .storage_rw(7, &output_buffer)
            .uniform(8, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![seq_len, batch_size, hidden_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bi_lstm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // Create test tensors: [seq_len=2, batch=1, input_size=3]
        let input = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 1, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let hidden_size = 4;
        let weight_ih = Tensor::from_vec_on(
            vec![0.1; 4 * hidden_size * 3], // [4*hidden, input]
            vec![4 * hidden_size, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_hh = Tensor::from_vec_on(
            vec![0.1; 4 * hidden_size * hidden_size], // [4*hidden, hidden]
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

        let result = BiLSTM::new(input, weight_ih, weight_hh, bias_ih, bias_hh, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[2, 1, 4]);
    }
}
