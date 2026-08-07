// SPDX-License-Identifier: AGPL-3.0-or-later
//! SGDW - SGD with Decoupled Weight Decay (Pure WGSL)
//!
//! More principled weight decay than L2 regularization
//! Decouples weight decay from gradient-based update
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// SGD with Decoupled Weight Decay
pub struct SGDW {
    parameters: Tensor,
    gradients: Tensor,
    velocity: Option<Tensor>,
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    dampening: f32,
    nesterov: bool,
}

impl SGDW {
    /// Create SGD with decoupled weight decay.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes mismatch or `learning_rate` <= 0.
    pub fn new(
        parameters: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        dampening: f32,
        nesterov: bool,
        velocity: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if parameters.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                parameters.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate learning rate is positive
        if learning_rate <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "sgdw",
                "learning_rate must be positive",
            ));
        }

        // Validate velocity shape if provided
        if let Some(ref v_tensor) = velocity
            && v_tensor.shape() != parameters.shape()
        {
            return Err(BarracudaError::shape_mismatch(
                v_tensor.shape().to_vec(),
                parameters.shape().to_vec(),
            ));
        }

        Ok(Self {
            parameters,
            gradients,
            velocity,
            learning_rate,
            momentum,
            weight_decay,
            dampening,
            nesterov,
        })
    }

    /// Execute SGDW optimizer step.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.parameters.device();
        let size = self.parameters.shape().iter().product::<usize>();
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;

        // Create writable buffers using GPU copy operations (zero CPU fallbacks)
        let parameters_buffer = device.create_buffer_f32(size)?;

        // Copy parameters buffer using GPU copy
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SGDW Buffer Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            self.parameters.buffer(),
            0,
            &parameters_buffer,
            0,
            byte_size,
        );

        // Create velocity buffer (GPU copy or zero initialization)
        let v_buffer = if let Some(ref v_tensor) = self.velocity {
            let v_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(v_tensor.buffer(), 0, &v_buf, 0, byte_size);
            v_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Submit buffer copies
        device.submit_commands(Some(encoder.finish()));

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            learning_rate: f32,
            momentum: f32,
            weight_decay: f32,
            dampening: f32,
            nesterov: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            learning_rate: self.learning_rate,
            momentum: self.momentum,
            weight_decay: self.weight_decay,
            dampening: self.dampening,
            nesterov: u32::from(self.nesterov),
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SGDW Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "SGDW")
            .shader(
                include_str!("../shaders/optimizer/sgdw_f64.wgsl"),
                "main",
            )
            .uniform(0, &params_buffer)
            .storage_read(1, &parameters_buffer)
            .storage_read(2, self.gradients.buffer())
            .storage_rw(3, &v_buffer)
            .storage_rw(4, &output_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_params = Tensor::from_buffer(
            output_buffer,
            self.parameters.shape().to_vec(),
            device.clone(),
        );

        let updated_v =
            Tensor::from_buffer(v_buffer, self.parameters.shape().to_vec(), device.clone());

        Ok((updated_params, updated_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sgdw_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let sgdw = SGDW::new(params, gradients, 0.01, 0.9, 0.0001, 0.0, false, None).unwrap();
        let (updated_params, _v) = sgdw.execute().unwrap();

        assert_eq!(updated_params.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_sgdw_with_momentum() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        // Step 1
        let sgdw1 = SGDW::new(
            params.clone(),
            gradients.clone(),
            0.01,
            0.9,
            0.0001,
            0.0,
            false,
            None,
        )
        .unwrap();
        let (params1, v1) = sgdw1.execute().unwrap();

        // Step 2 with velocity
        let sgdw2 = SGDW::new(params1, gradients, 0.01, 0.9, 0.0001, 0.0, false, Some(v1)).unwrap();
        let (params2, _v2) = sgdw2.execute().unwrap();

        assert_eq!(params2.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_sgdw_nesterov() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        let sgdw = SGDW::new(params, gradients, 0.01, 0.9, 0.0001, 0.0, true, None).unwrap();
        let (updated_params, _v) = sgdw.execute().unwrap();

        assert_eq!(updated_params.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_sgdw_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 128;
        let params = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
            .await
            .unwrap();

        let sgdw = SGDW::new(params, gradients, 0.01, 0.9, 0.0001, 0.0, false, None).unwrap();
        let (updated_params, updated_v) = sgdw.execute().unwrap();

        assert_eq!(updated_params.shape(), &[size]);
        assert_eq!(updated_v.shape(), &[size]);
    }
}
