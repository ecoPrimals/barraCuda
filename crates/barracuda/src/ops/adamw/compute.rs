// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for `AdamW` Optimizer
//!
//! This module contains the single-pass GPU execution for `AdamW` optimizer
//! with decoupled weight decay.

use super::{AdamW, AdamWParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl AdamW {
    /// Execute `AdamW` optimizer step (GPU single-pass with decoupled weight decay)
    ///
    /// **Deep Debt**: Efficient single-pass update with decoupled weight decay
    ///
    /// Returns: (`new_params`, `new_m`, `new_v`)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.params().device();
        let size = self.params().len();

        // Create parameters
        let params = AdamWParams {
            num_params: size as u32,
            learning_rate: self.learning_rate(),
            beta1: self.beta1(),
            beta2: self.beta2(),
            epsilon: self.epsilon(),
            weight_decay: self.weight_decay(),
            step: self.step(),
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AdamW Params"),
            size: std::mem::size_of::<AdamWParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Output buffers
        let params_out_buffer = device.create_buffer_f32(size)?;
        let m_out_buffer = device.create_buffer_f32(size)?;
        let v_out_buffer = device.create_buffer_f32(size)?;

        // Copy initial params, m, v to output buffers (will be updated in-place)
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("AdamW Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            self.params().buffer(),
            0,
            &params_out_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            self.m().buffer(),
            0,
            &m_out_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            self.v().buffer(),
            0,
            &v_out_buffer,
            0,
            (size * std::mem::size_of::<f32>()) as u64,
        );
        device.submit_commands(Some(encoder.finish()));

        ComputeDispatch::new(device, "AdamW")
            .shader(Self::shader(), "main")
            .storage_read(0, self.gradients().buffer())
            .storage_rw(1, &params_out_buffer)
            .storage_rw(2, &m_out_buffer)
            .storage_rw(3, &v_out_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Return all three outputs
        Ok((
            Tensor::from_buffer(
                params_out_buffer,
                self.params().shape().to_vec(),
                device.clone(),
            ),
            Tensor::from_buffer(m_out_buffer, self.m().shape().to_vec(), device.clone()),
            Tensor::from_buffer(v_out_buffer, self.v().shape().to_vec(), device.clone()),
        ))
    }
}
