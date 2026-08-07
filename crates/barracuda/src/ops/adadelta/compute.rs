// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for `AdaDelta` Optimizer
//!
//! This module contains the GPU execution for `AdaDelta` optimizer
//! with adaptive learning rate.

use super::{AdaDelta, AdaDeltaParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl AdaDelta {
    /// Execute `AdaDelta` optimizer step (GPU single-pass)
    /// **Deep Debt**: Efficient single-pass update with adaptive learning rate
    /// Returns: (`updated_weights`, `updated_acc_grad`, `updated_acc_delta`)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.weights().device();
        let size = self.weights().shape().iter().product::<usize>();

        let params = AdaDeltaParams {
            rho: self.rho(),
            epsilon: 1e-6,
            weight_decay: 0.0,
            _padding: 0,
        };

        // Create state buffers if not provided
        let zeros = vec![0.0f32; size];
        let acc_grad_in = if let Some(tensor) = self.acc_grad() {
            tensor.buffer()
        } else {
            &device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adadelta_acc_grad_zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let acc_delta_in = if let Some(tensor) = self.acc_delta() {
            tensor.buffer()
        } else {
            &device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adadelta_acc_delta_zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let weights_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adadelta_weights_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let acc_grad_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adadelta_acc_grad_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let acc_delta_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adadelta_acc_delta_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adadelta_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "AdaDelta")
            .shader(Self::shader(), "main")
            .storage_read(0, self.weights().buffer())
            .storage_read(1, self.gradients().buffer())
            .storage_read(2, acc_grad_in)
            .storage_read(3, acc_delta_in)
            .storage_rw(4, &weights_out_buffer)
            .storage_rw(5, &acc_grad_out_buffer)
            .storage_rw(6, &acc_delta_out_buffer)
            .uniform(7, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_weights = Tensor::from_buffer(
            weights_out_buffer,
            self.weights().shape().to_vec(),
            device.clone(),
        );

        let updated_acc_grad = Tensor::from_buffer(
            acc_grad_out_buffer,
            self.weights().shape().to_vec(),
            device.clone(),
        );

        let updated_acc_delta = Tensor::from_buffer(
            acc_delta_out_buffer,
            self.weights().shape().to_vec(),
            device.clone(),
        );

        Ok((updated_weights, updated_acc_grad, updated_acc_delta))
    }
}
