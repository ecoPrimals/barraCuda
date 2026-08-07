// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PoissonNLLLoss` - Pure WGSL
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

/// Poisson Negative Log Likelihood Loss operation
pub struct PoissonNLLLoss {
    input: Tensor,
    target: Tensor,
    log_input: bool,
    full: bool,
    epsilon: f32,
}

impl PoissonNLLLoss {
    /// Create a new Poisson NLL loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if input and target shapes do not match.
    pub fn new(
        input: Tensor,
        target: Tensor,
        log_input: Option<bool>,
        full: Option<bool>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        if input.shape() != target.shape() {
            return Err(BarracudaError::invalid_op(
                "poisson_nll_loss",
                "input and target shapes must match",
            ));
        }

        Ok(Self {
            input,
            target,
            log_input: log_input.unwrap_or(false),
            full: full.unwrap_or(false),
            epsilon: epsilon.unwrap_or(1e-8),
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        const SHADER: &str = include_str!("../shaders/loss/poisson_nll_loss_f64.wgsl");
        SHADER
    }

    /// Execute the Poisson NLL loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let size = self.input.shape().iter().product::<usize>();
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            log_input: u32,
            full: u32,
            epsilon: f32,
        }

        let params = Params {
            size: size as u32,
            log_input: u32::from(self.log_input),
            full: u32::from(self.full),
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PoissonNLLLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "PoissonNLLLoss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.target.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_poisson_nll_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 10;

        let input = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let target = Tensor::from_vec_on(vec![2.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let output = PoissonNLLLoss::new(input, target, None, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[size]);
    }
}
