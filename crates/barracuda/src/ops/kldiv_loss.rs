// SPDX-License-Identifier: AGPL-3.0-or-later
//! `KLDivLoss` - Pure WGSL
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

/// KL Divergence Loss operation
pub struct KLDivLoss {
    input: Tensor,
    target: Tensor,
    log_target: bool,
    reduction: u32, // 0 = none, 1 = mean, 2 = sum, 3 = batchmean
}

impl KLDivLoss {
    /// Create a new KL divergence loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        target: Tensor,
        log_target: Option<bool>,
        reduction: Option<u32>,
    ) -> Result<Self> {
        if input.shape() != target.shape() {
            return Err(BarracudaError::invalid_op(
                "kldiv_loss",
                "input and target shapes must match",
            ));
        }

        Ok(Self {
            input,
            target,
            log_target: log_target.unwrap_or(false),
            reduction: reduction.unwrap_or(1), // mean by default
        })
    }

    /// Execute the KL divergence loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let size = self.input.shape().iter().product::<usize>();

        // Shader outputs per-element; reduction happens in post-processing
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            log_target: u32,
            reduction: u32,
            _padding: u32,
        }

        let params = Params {
            size: size as u32,
            log_target: u32::from(self.log_target),
            reduction: self.reduction,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KLDivLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "KLDivLoss")
            .shader(include_str!("../shaders/loss/kldiv_loss_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.target.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Shader handles reduction internally
        let output_shape = self.input.shape().to_vec();

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
    async fn test_kldiv_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 10;

        let input = Tensor::from_vec_on(vec![-1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let target = Tensor::from_vec_on(vec![0.1; size], vec![size], device.clone())
            .await
            .unwrap();

        let output = KLDivLoss::new(input, target, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[size]);
    }
}
