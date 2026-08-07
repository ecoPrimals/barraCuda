// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU compute operations for Adam Optimizer
//!
//! This module contains the single-pass GPU execution for Adam optimizer
//! with bias correction.

use super::{Adam, AdamParams};
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

impl Adam {
    /// Execute Adam optimizer step (GPU single-pass with bias correction)
    ///
    /// **Deep Debt**: Efficient single-pass update with bias correction
    ///
    /// Returns: (`new_params`, `new_m`, `new_v`)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.params().device();
        let size = self.params().shape().iter().product::<usize>();

        let adam_params = AdamParams {
            num_params: size as u32,
            learning_rate: self.learning_rate(),
            beta1: self.beta1(),
            beta2: self.beta2(),
            epsilon: 1e-8,
            weight_decay: 0.0,
            step: self.step() as u32,
        };

        // Create writable buffers (shader does in-place updates)
        let zeros = vec![0.0f32; size];

        // Copy params to writable buffer
        let params_data = self.params().to_vec()?;
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adam_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Copy or create m buffer
        let m_data = if let Some(m_tensor) = self.m() {
            m_tensor.to_vec()?
        } else {
            zeros.clone()
        };
        let m_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adam_m"),
                contents: bytemuck::cast_slice(&m_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Copy or create v buffer
        let v_data = if let Some(v_tensor) = self.v() {
            v_tensor.to_vec()?
        } else {
            zeros
        };
        let v_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adam_v"),
                contents: bytemuck::cast_slice(&v_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let adam_params_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adam_params"),
                    contents: bytemuck::cast_slice(&[adam_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        ComputeDispatch::new(device, "Adam")
            .shader(Self::shader(), "main")
            .storage_read(0, self.gradients().buffer())
            .storage_rw(1, &params_buffer)
            .storage_rw(2, &m_buffer)
            .storage_rw(3, &v_buffer)
            .uniform(4, &adam_params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_params = Tensor::from_buffer(
            params_buffer,
            self.params().shape().to_vec(),
            device.clone(),
        );

        let updated_m =
            Tensor::from_buffer(m_buffer, self.params().shape().to_vec(), device.clone());

        let updated_v =
            Tensor::from_buffer(v_buffer, self.params().shape().to_vec(), device.clone());

        Ok((updated_params, updated_m, updated_v))
    }
}
