// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MultiLabelMarginLoss` - Pure WGSL
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

/// Multi-Label Margin Loss operation
pub struct MultiLabelMarginLoss {
    input: Tensor,
    target: Tensor,
    num_classes: usize,
}

impl MultiLabelMarginLoss {
    /// Create a new multi-label margin loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, target: Tensor) -> Result<Self> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_classes = input_shape[1..].iter().product::<usize>();

        let target_shape = target.shape();
        if target_shape[0] != batch_size
            || target_shape[1..].iter().product::<usize>() != input_classes
        {
            return Err(BarracudaError::invalid_op(
                "multilabel_margin_loss",
                "target shape must match input shape",
            ));
        }

        Ok(Self {
            input,
            target,
            num_classes: input_classes,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/math/multilabel_margin_loss_f64.wgsl")
    }

    /// Execute the multi-label margin loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let batch_size = self.input.shape()[0];
        let output_size = batch_size;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            num_classes: u32,
            _padding: [u32; 2],
        }

        let params = Params {
            batch_size: batch_size as u32,
            num_classes: self.num_classes as u32,
            _padding: [0, 0],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MultiLabelMarginLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "MultiLabelMarginLoss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.target.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_multilabel_margin_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 2;
        let num_classes = 3;

        let input = Tensor::from_vec_on(
            vec![0.9, 0.1, 0.1, 0.1, 0.8, 0.2],
            vec![batch_size, num_classes],
            device.clone(),
        )
        .await
        .unwrap();

        let target = Tensor::from_vec_on(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            vec![batch_size, num_classes],
            device.clone(),
        )
        .await
        .unwrap();

        let output = MultiLabelMarginLoss::new(input, target)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[batch_size]);
    }
}
