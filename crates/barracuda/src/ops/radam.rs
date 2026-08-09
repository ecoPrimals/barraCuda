// SPDX-License-Identifier: AGPL-3.0-or-later
//! `RAdam` - Rectified Adam Optimizer (Pure WGSL)
//!
//! Addresses variance warmup issue in Adam
//! Automatically adjusts learning rate based on variance tractability
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Rectified Adam Optimizer
pub struct RAdam {
    parameters: Tensor,
    gradients: Tensor,
    momentum: Option<Tensor>,
    variance: Option<Tensor>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    step: usize,
}

impl RAdam {
    /// Create an `RAdam` optimizer step with the given parameters and optional momentum/variance.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes mismatch, `learning_rate` <= 0, betas not in [0, 1), or step == 0.
    pub fn new(
        parameters: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        step: usize,
        momentum: Option<Tensor>,
        variance: Option<Tensor>,
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
                "radam",
                "learning_rate must be positive",
            ));
        }

        // Validate betas in valid range
        if !(0.0..1.0).contains(&beta1) {
            return Err(BarracudaError::invalid_op(
                "radam",
                "beta1 must be in range [0.0, 1.0)",
            ));
        }

        if !(0.0..1.0).contains(&beta2) {
            return Err(BarracudaError::invalid_op(
                "radam",
                "beta2 must be in range [0.0, 1.0)",
            ));
        }

        // Validate step is positive
        if step == 0 {
            return Err(BarracudaError::invalid_op(
                "radam",
                "step must be >= 1 (starts at 1, not 0)",
            ));
        }

        // Validate momentum and variance shapes if provided
        if let Some(ref m_tensor) = momentum
            && m_tensor.shape() != parameters.shape()
        {
            return Err(BarracudaError::shape_mismatch(
                m_tensor.shape().to_vec(),
                parameters.shape().to_vec(),
            ));
        }

        if let Some(ref v_tensor) = variance
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
            momentum,
            variance,
            learning_rate,
            beta1,
            beta2,
            step,
        })
    }

    /// Execute one `RAdam` optimization step. Returns (`updated_params`, `updated_momentum`, `updated_variance`).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.parameters.device();
        let size = self.parameters.shape().iter().product::<usize>();
        let byte_size = (size * std::mem::size_of::<f32>()) as u64;

        // Create writable buffers using GPU copy operations (zero CPU fallbacks)
        let parameters_buffer = device.create_buffer_f32(size)?;

        // Copy parameters buffer using GPU copy
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RAdam Buffer Copy Encoder"),
        });
        encoder.copy_buffer_to_buffer(
            self.parameters.buffer(),
            0,
            &parameters_buffer,
            0,
            byte_size,
        );

        // Create momentum buffer (GPU copy or zero initialization)
        let m_buffer = if let Some(ref m_tensor) = self.momentum {
            let m_buf = device.create_buffer_f32(size)?;
            encoder.copy_buffer_to_buffer(m_tensor.buffer(), 0, &m_buf, 0, byte_size);
            m_buf
        } else {
            device.create_buffer_f32(size)?
        };

        // Create variance buffer (GPU copy or zero initialization)
        let v_buffer = if let Some(ref v_tensor) = self.variance {
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
            step: u32,
            learning_rate: f32,
            beta1: f32,
            beta2: f32,
            epsilon: f32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            size: size as u32,
            step: self.step as u32,
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: 1e-8,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RAdam Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "RAdam")
            .shader(include_str!("../shaders/optimizer/radam_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, &parameters_buffer)
            .storage_read(2, self.gradients.buffer())
            .storage_rw(3, &m_buffer)
            .storage_rw(4, &v_buffer)
            .storage_rw(5, &output_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_params = Tensor::from_buffer(
            output_buffer,
            self.parameters.shape().to_vec(),
            device.clone(),
        );

        let updated_m =
            Tensor::from_buffer(m_buffer, self.parameters.shape().to_vec(), device.clone());
        let updated_v =
            Tensor::from_buffer(v_buffer, self.parameters.shape().to_vec(), device.clone());

        Ok((updated_params, updated_m, updated_v))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_radam_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let radam = RAdam::new(params, gradients, 0.001, 0.9, 0.999, 1, None, None).unwrap();
        let (updated_params, _m, _v) = radam.execute().unwrap();

        assert_eq!(updated_params.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_radam_with_state() {
        let device = crate::device::test_pool::get_test_device().await;
        let params = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        // Step 1
        let radam1 = RAdam::new(
            params.clone(),
            gradients.clone(),
            0.001,
            0.9,
            0.999,
            1,
            None,
            None,
        )
        .unwrap();
        let (params1, m1, v1) = radam1.execute().unwrap();

        // Step 2 with accumulated state
        let radam2 =
            RAdam::new(params1, gradients, 0.001, 0.9, 0.999, 2, Some(m1), Some(v1)).unwrap();
        let (params2, _m2, _v2) = radam2.execute().unwrap();

        assert_eq!(params2.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_radam_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 128;
        let params = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
            .await
            .unwrap();

        let radam = RAdam::new(params, gradients, 0.001, 0.9, 0.999, 1, None, None).unwrap();
        let (updated_params, updated_m, updated_v) = radam.execute().unwrap();

        assert_eq!(updated_params.shape(), &[size]);
        assert_eq!(updated_m.shape(), &[size]);
        assert_eq!(updated_v.shape(), &[size]);
    }
}
