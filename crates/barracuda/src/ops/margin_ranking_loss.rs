// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MarginRankingLoss` - Pure WGSL
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

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/loss/margin_ranking_loss_f64.wgsl");

/// Margin Ranking Loss operation
pub struct MarginRankingLoss {
    input1: Tensor,
    input2: Tensor,
    target: Tensor,
    margin: f32,
}

impl MarginRankingLoss {
    /// Create a new margin ranking loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input1: Tensor, input2: Tensor, target: Tensor, margin: f32) -> Result<Self> {
        if input1.shape() != input2.shape() {
            return Err(BarracudaError::invalid_op(
                "margin_ranking_loss",
                "input1 and input2 shapes must match",
            ));
        }

        let batch_size = input1.shape()[0];
        if target.shape()[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "margin_ranking_loss",
                "target batch size mismatch",
            ));
        }

        Ok(Self {
            input1,
            input2,
            target,
            margin,
        })
    }

    /// Execute the margin ranking loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input1.device();

        let size = self.input1.shape().iter().product::<usize>();
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            margin: f32,
            _padding: [u32; 2],
        }

        let params = Params {
            size: size as u32,
            margin: self.margin,
            _padding: [0, 0],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MarginRankingLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "MarginRankingLoss")
            .shader(SHADER_F64, "main")
            .storage_read(0, self.input1.buffer())
            .storage_read(1, self.input2.buffer())
            .storage_read(2, self.target.buffer())
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input1.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_margin_ranking_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 10;

        let input1 = Tensor::from_vec_on(vec![2.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let input2 = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let target = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let output = MarginRankingLoss::new(input1, input2, target, 0.5)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[size]);
    }
}
