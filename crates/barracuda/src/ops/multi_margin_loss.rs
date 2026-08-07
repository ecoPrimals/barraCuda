// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MultiMarginLoss` - Pure WGSL
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

/// Multi-Margin Loss operation
pub struct MultiMarginLoss {
    input: Tensor,
    target: Tensor,
    weight: Option<Tensor>,
    num_classes: usize,
    p: u32,
    margin: f32,
}

impl MultiMarginLoss {
    /// Create a new multi-margin loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        target: Tensor,
        weight: Option<Tensor>,
        num_classes: usize,
        p: u32,
        margin: f32,
    ) -> Result<Self> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_classes = input_shape[1..].iter().product::<usize>();

        if input_classes != num_classes {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "input must have num_classes columns",
            ));
        }

        let target_shape = target.shape();
        if target_shape[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "target batch size mismatch",
            ));
        }

        if let Some(ref w) = weight
            && w.shape().iter().product::<usize>() != num_classes
        {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "weight must have num_classes elements",
            ));
        }

        if p != 1 && p != 2 {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "p must be 1 or 2",
            ));
        }

        Ok(Self {
            input,
            target,
            weight,
            num_classes,
            p,
            margin,
        })
    }

    /// Execute the multi-margin loss operation
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
            p: u32,
            margin: f32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            num_classes: self.num_classes as u32,
            p: self.p,
            margin: self.margin,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MultiMarginLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create weight buffer (use ones if not provided)
        let ones_buffer;
        let weight_buffer = if let Some(ref w) = self.weight {
            w.buffer()
        } else {
            let ones = vec![1.0f32; self.num_classes];
            ones_buffer = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("MultiMarginLoss Weight"),
                    contents: bytemuck::cast_slice(&ones),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            &ones_buffer
        };

        ComputeDispatch::new(device, "MultiMarginLoss")
            .shader(
                include_str!("../shaders/math/multi_margin_loss_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.target.buffer())
            .storage_read(2, weight_buffer)
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
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
    async fn test_multi_margin_loss_basic() {
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

        let target = Tensor::from_vec_on(vec![0.0f32, 1.0f32], vec![batch_size], device.clone())
            .await
            .unwrap();

        let output = MultiMarginLoss::new(input, target, None, num_classes, 1, 1.0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[batch_size]);
    }
}
