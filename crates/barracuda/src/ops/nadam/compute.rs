// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for `NAdam` Optimizer
//!
//! This module contains the single-pass GPU execution for `NAdam` optimizer
//! with Nesterov momentum.

use super::{Nadam, NadamParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl Nadam {
    /// Execute `NAdam` optimizer step (GPU single-pass)
    ///
    /// **Deep Debt**: Efficient single-pass update with Nesterov momentum
    ///
    /// Returns: (`new_weights`, `new_m`, `new_v`)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.weights().device();
        let size = self.weights().len();

        // Create parameters
        let params = NadamParams {
            learning_rate: self.learning_rate(),
            beta1: self.beta1(),
            beta2: self.beta2(),
            epsilon: self.epsilon(),
            weight_decay: self.weight_decay(),
            step: self.step(),
            _padding: [0, 0],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("NAdam Params"),
            size: std::mem::size_of::<NadamParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Output buffers
        let weights_out_buffer = device.create_buffer_f32(size)?;
        let m_out_buffer = device.create_buffer_f32(size)?;
        let v_out_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "NAdam")
            .shader(Self::shader(), "main")
            .storage_read(0, self.weights().buffer())
            .storage_read(1, self.gradients().buffer())
            .storage_read(2, self.m().buffer())
            .storage_read(3, self.v().buffer())
            .storage_rw(4, &weights_out_buffer)
            .storage_rw(5, &m_out_buffer)
            .storage_rw(6, &v_out_buffer)
            .uniform(7, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok((
            Tensor::from_buffer(
                weights_out_buffer,
                self.weights().shape().to_vec(),
                device.clone(),
            ),
            Tensor::from_buffer(m_out_buffer, self.m().shape().to_vec(), device.clone()),
            Tensor::from_buffer(v_out_buffer, self.v().shape().to_vec(), device.clone()),
        ))
    }
}
