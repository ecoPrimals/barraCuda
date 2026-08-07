// SPDX-License-Identifier: AGPL-3.0-or-later
//! `WassersteinLoss` - Pure WGSL
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

/// Wasserstein Loss operation
pub struct WassersteinLoss {
    pred: Tensor,
    target: Tensor,
}

impl WassersteinLoss {
    /// Create a new Wasserstein loss operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(pred: Tensor, target: Tensor) -> Result<Self> {
        if pred.shape() != target.shape() {
            return Err(BarracudaError::invalid_op(
                "wasserstein_loss",
                "pred and target shapes must match",
            ));
        }

        Ok(Self { pred, target })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        const SHADER: &str = include_str!("../shaders/loss/wasserstein_loss_f64.wgsl");
        SHADER
    }

    /// Execute the Wasserstein loss operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.pred.device();

        let size = self.pred.shape().iter().product::<usize>();

        // Output is scalar distance
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            _padding: [u32; 3],
        }

        let params = Params {
            size: size as u32,
            _padding: [0, 0, 0],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WassersteinLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "WassersteinLoss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.pred.buffer())
            .storage_read(1, self.target.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;
        Ok(Tensor::new(output_data, vec![size], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wasserstein_loss_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 10;

        let pred = Tensor::from_vec_on(vec![0.1; size], vec![size], device.clone())
            .await
            .unwrap();

        let target = Tensor::from_vec_on(vec![0.1; size], vec![size], device.clone())
            .await
            .unwrap();

        let output = WassersteinLoss::new(pred, target)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[size]);
    }
}
