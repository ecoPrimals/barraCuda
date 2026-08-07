// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for SGD Optimizer
//!
//! This module contains the GPU execution for SGD optimizer
//! with optional momentum and weight decay.

use super::{SGD, SGDParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl SGD {
    /// Execute SGD optimizer step (GPU execution)
    /// **Deep Debt**: Efficient GPU update with optional momentum and weight decay
    /// Returns: (`updated_weights`, `updated_velocity`)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Option<Tensor>)> {
        let device = self.weights().device();
        let size = self.weights().shape().iter().product::<usize>();

        let params = SGDParams {
            learning_rate: self.learning_rate(),
            momentum: self.momentum(),
            weight_decay: self.weight_decay(),
            dampening: 0.0,
        };

        // Create velocity buffer if not provided
        let velocity_in = if let Some(v) = self.velocity() {
            v.buffer()
        } else {
            let zeros = vec![0.0f32; size];
            &device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sgd_velocity_zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let weights_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sgd_weights_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let velocity_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sgd_velocity_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sgd_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "SGD")
            .shader(Self::shader(), "main")
            .storage_read(0, self.weights().buffer())
            .storage_read(1, self.gradients().buffer())
            .storage_read(2, velocity_in)
            .storage_rw(3, &weights_out_buffer)
            .storage_rw(4, &velocity_out_buffer)
            .uniform(5, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_weights = Tensor::from_buffer(
            weights_out_buffer,
            self.weights().shape().to_vec(),
            device.clone(),
        );

        let updated_velocity = if self.momentum() == 0.0 {
            None
        } else {
            Some(Tensor::from_buffer(
                velocity_out_buffer,
                self.weights().shape().to_vec(),
                device.clone(),
            ))
        };

        Ok((updated_weights, updated_velocity))
    }
}
