// SPDX-License-Identifier: AGPL-3.0-or-later
//! RNN Cell - Basic recurrent neural network cell
//!
//! **Canonical `BarraCuda` Pattern**: Struct with new/execute
//!
//! ## Algorithm
//!
//! ```text
//! h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// RNN cell weight matrices and biases.
#[derive(Clone)]
pub struct RNNWeights {
    /// Input-to-hidden weights (`hidden_size` × `input_size`).
    pub w_ih: Vec<f32>,
    /// Hidden-to-hidden weights (`hidden_size` × `hidden_size`).
    pub w_hh: Vec<f32>,
    /// Input bias (`hidden_size`).
    pub b_ih: Vec<f32>,
    /// Hidden bias (`hidden_size`).
    pub b_hh: Vec<f32>,
}

/// RNN cell operation (`h_t` = `tanh(W_ih` x + `b_ih` + `W_hh` h + `b_hh`)).
pub struct RNNCell {
    input: Tensor,
    prev_hidden: Tensor,
    weights: RNNWeights,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
}

impl RNNCell {
    /// Create a new RNN cell operation.
    /// # Errors
    /// Returns [`Err`] if input or hidden shapes are invalid, or if weight/bias
    /// dimensions do not match the expected sizes.
    pub fn new(
        input: Tensor,
        prev_hidden: Tensor,
        weights: RNNWeights,
        batch_size: usize,
        input_size: usize,
        hidden_size: usize,
    ) -> Result<Self> {
        // Validate input shapes
        let input_shape = input.shape();
        let hidden_shape = prev_hidden.shape();

        if input_shape.len() != 2 || input_shape[0] != batch_size || input_shape[1] != input_size {
            return Err(BarracudaError::InvalidShape {
                expected: vec![batch_size, input_size],
                actual: input_shape.to_vec(),
            });
        }

        if hidden_shape.len() != 2
            || hidden_shape[0] != batch_size
            || hidden_shape[1] != hidden_size
        {
            return Err(BarracudaError::InvalidShape {
                expected: vec![batch_size, hidden_size],
                actual: hidden_shape.to_vec(),
            });
        }

        // Validate weight dimensions
        let expected_w_ih_size = hidden_size * input_size;
        let expected_w_hh_size = hidden_size * hidden_size;

        if weights.w_ih.len() != expected_w_ih_size {
            return Err(BarracudaError::invalid_input(format!(
                "w_ih size mismatch: expected {}, got {}",
                expected_w_ih_size,
                weights.w_ih.len()
            )));
        }

        if weights.w_hh.len() != expected_w_hh_size {
            return Err(BarracudaError::invalid_input(format!(
                "w_hh size mismatch: expected {}, got {}",
                expected_w_hh_size,
                weights.w_hh.len()
            )));
        }

        if weights.b_ih.len() != hidden_size {
            return Err(BarracudaError::invalid_input(format!(
                "b_ih size mismatch: expected {}, got {}",
                hidden_size,
                weights.b_ih.len()
            )));
        }

        if weights.b_hh.len() != hidden_size {
            return Err(BarracudaError::invalid_input(format!(
                "b_hh size mismatch: expected {}, got {}",
                hidden_size,
                weights.b_hh.len()
            )));
        }

        Ok(Self {
            input,
            prev_hidden,
            weights,
            batch_size,
            input_size,
            hidden_size,
        })
    }

    /// Execute the RNN cell and return the new hidden state.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let output_size = self.batch_size * self.hidden_size;

        // Create buffers for weights and biases
        let w_ih_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RNN w_ih"),
                contents: bytemuck::cast_slice(&self.weights.w_ih),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let w_hh_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RNN w_hh"),
                contents: bytemuck::cast_slice(&self.weights.w_hh),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let b_ih_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RNN b_ih"),
                contents: bytemuck::cast_slice(&self.weights.b_ih),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let b_hh_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RNN b_hh"),
                contents: bytemuck::cast_slice(&self.weights.b_hh),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

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
                label: Some("RNN Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "RNN Cell")
            .shader(include_str!("../shaders/rnn/rnn_cell_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, &w_ih_buffer)
            .storage_read(2, &w_hh_buffer)
            .storage_read(3, &b_ih_buffer)
            .storage_read(4, &b_hh_buffer)
            .storage_read(5, self.prev_hidden.buffer())
            .storage_rw(6, &output_buffer)
            .uniform(7, &params_buffer)
            .dispatch_1d(self.batch_size as u32)
            .submit()?;

        // Output shape: [batch_size, hidden_size]
        let output_shape = vec![self.batch_size, self.hidden_size];

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
    async fn test_rnn_cell_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![0.5; 2 * 4], vec![2, 4], device.clone())
            .await
            .unwrap();
        let prev_hidden = Tensor::from_vec_on(vec![0.0; 2 * 8], vec![2, 8], device.clone())
            .await
            .unwrap();
        let weights = RNNWeights {
            w_ih: vec![0.01; 8 * 4],
            w_hh: vec![0.01; 8 * 8],
            b_ih: vec![0.0; 8],
            b_hh: vec![0.0; 8],
        };
        let hidden = RNNCell::new(input, prev_hidden, weights, 2, 4, 8)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(hidden.shape(), &[2, 8]);
        let result = hidden.to_vec().unwrap();
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_rnn_cell_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Single batch
        let input = Tensor::from_vec_on(vec![1.0; 3], vec![1, 3], device.clone())
            .await
            .unwrap();
        let prev_hidden = Tensor::from_vec_on(vec![0.0; 4], vec![1, 4], device.clone())
            .await
            .unwrap();
        let weights = RNNWeights {
            w_ih: vec![0.1; 4 * 3],
            w_hh: vec![0.1; 4 * 4],
            b_ih: vec![0.0; 4],
            b_hh: vec![0.0; 4],
        };
        let hidden = RNNCell::new(input, prev_hidden, weights, 1, 3, 4)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(hidden.shape(), &[1, 4]);
    }

    #[tokio::test]
    async fn test_rnn_cell_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Non-zero previous hidden state
        let input = Tensor::from_vec_on(vec![0.5; 4], vec![1, 4], device.clone())
            .await
            .unwrap();
        let prev_hidden = Tensor::from_vec_on(vec![0.5; 8], vec![1, 8], device.clone())
            .await
            .unwrap();
        let weights = RNNWeights {
            w_ih: vec![0.1; 8 * 4],
            w_hh: vec![0.1; 8 * 8],
            b_ih: vec![0.1; 8],
            b_hh: vec![0.1; 8],
        };
        let hidden = RNNCell::new(input, prev_hidden, weights, 1, 4, 8)
            .unwrap()
            .execute()
            .unwrap();
        let result = hidden.to_vec().unwrap();
        assert!(result.iter().all(|&x| x.is_finite()));
        // tanh bounds: -1 < x < 1
        assert!(result.iter().all(|&x| x > -1.0 && x < 1.0));
    }
}
