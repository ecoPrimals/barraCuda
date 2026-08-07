// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for `RMSprop` Optimizer
//!
//! This module contains the GPU execution for `RMSprop` optimizer
//! with adaptive learning rate per parameter.

use super::{RMSprop, RMSpropParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl RMSprop {
    /// Execute `RMSprop` optimizer step (GPU single-pass)
    /// **Deep Debt**: Efficient single-pass update with adaptive learning rate
    /// Returns: (`updated_weights`, `updated_sq_avg`)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.weights().device();
        let size = self.weights().shape().iter().product::<usize>();

        let params = RMSpropParams {
            learning_rate: self.learning_rate(),
            alpha: self.alpha(),
            epsilon: 1e-8,
            weight_decay: 0.0,
        };

        // Create sq_avg buffer if not provided
        let sq_avg_in = if let Some(sq) = self.sq_avg() {
            sq.buffer()
        } else {
            let zeros = vec![0.0f32; size];
            &device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("rmsprop_sq_avg_zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let weights_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmsprop_weights_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let sq_avg_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rmsprop_sq_avg_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("rmsprop_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "RMSprop")
            .shader(Self::shader(), "main")
            .storage_read(0, self.weights().buffer())
            .storage_read(1, self.gradients().buffer())
            .storage_read(2, sq_avg_in)
            .storage_rw(3, &weights_out_buffer)
            .storage_rw(4, &sq_avg_out_buffer)
            .uniform(5, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_weights = Tensor::from_buffer(
            weights_out_buffer,
            self.weights().shape().to_vec(),
            device.clone(),
        );

        let updated_sq_avg = Tensor::from_buffer(
            sq_avg_out_buffer,
            self.weights().shape().to_vec(),
            device.clone(),
        );

        Ok((updated_weights, updated_sq_avg))
    }
}
