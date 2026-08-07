// SPDX-License-Identifier: AGPL-3.0-or-later
//! `AdaGrad` Optimizer - GPU-accelerated Adaptive Gradient Algorithm
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (existing shader evolved)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)
//!
//! ## Algorithm
//!
//! ```text
//! G_t = G_{t-1} + g_t²
//! w_t = w_{t-1} - (lr / sqrt(G_t + ε)) * g_t
//! ```
//!
//! **Key Properties**:
//! - Adapts learning rate per parameter based on gradient history
//! - Accumulates squared gradients
//! - Good for sparse gradients
//! - Learning rate monotonically decreases (limitation)
//!
//! **Parameters**:
//! - `learning_rate`: Initial step size, typically 0.01
//! - `epsilon` (ε): Numerical stability constant, typically 1e-8
//!
//! **Used By**: Sparse gradient problems, NLP tasks
//!
//! **Note**: `RMSprop` and Adam address `AdaGrad`'s monotonic decrease issue
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let weights = Tensor::randn(vec![1000]).await?;
//! let gradients = Tensor::randn(vec![1000]).await?;
//!
//! // First step
//! let (w1, acc1) = weights.adagrad_step(&gradients, 0.01, None)?;
//!
//! // Subsequent steps
//! let (w2, acc2) = w1.adagrad_step(&gradients, 0.01, Some(&acc1))?;
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaGradParams {
    learning_rate: f32,
    epsilon: f32,
    weight_decay: f32,
    _padding: u32,
}

/// `AdaGrad` optimizer — per-parameter adaptive learning rates.
pub struct AdaGrad {
    weights: Tensor,
    gradients: Tensor,
    accumulated: Option<Tensor>,
    learning_rate: f32,
}

impl AdaGrad {
    /// Create `AdaGrad` optimizer. Good for sparse gradients.
    /// # Errors
    /// Returns [`Err`] if weights and gradients shapes differ, `learning_rate` is not positive,
    /// or accumulated (if provided) shape differs from weights.
    pub fn new(
        weights: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        accumulated: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if weights.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                weights.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate learning rate is positive
        if learning_rate <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "adagrad",
                "learning_rate must be positive",
            ));
        }

        // Validate accumulated shape if provided
        if let Some(ref acc) = accumulated
            && acc.shape() != weights.shape()
        {
            return Err(BarracudaError::shape_mismatch(
                acc.shape().to_vec(),
                weights.shape().to_vec(),
            ));
        }

        Ok(Self {
            weights,
            gradients,
            accumulated,
            learning_rate,
        })
    }

    /// Execute `AdaGrad` step. Returns (`updated_weights`, accumulated).
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.weights.device();
        let size = self.weights.shape().iter().product::<usize>();

        let params = AdaGradParams {
            learning_rate: self.learning_rate,
            epsilon: 1e-8,
            weight_decay: 0.0,
            _padding: 0,
        };

        // Create accumulated buffer if not provided
        let zeros = vec![0.0f32; size];
        let accumulated_in = if let Some(ref acc_tensor) = self.accumulated {
            acc_tensor.buffer()
        } else {
            &device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adagrad_acc_zeros"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                })
        };

        let weights_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adagrad_weights_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let accumulated_out_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adagrad_accumulated_out"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adagrad_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "AdaGrad")
            .shader(
                include_str!("../shaders/optimizer/adagrad_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.weights.buffer())
            .storage_read(1, self.gradients.buffer())
            .storage_read(2, accumulated_in)
            .storage_rw(3, &weights_out_buffer)
            .storage_rw(4, &accumulated_out_buffer)
            .uniform(5, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        let updated_weights = Tensor::from_buffer(
            weights_out_buffer,
            self.weights.shape().to_vec(),
            device.clone(),
        );

        let updated_accumulated = Tensor::from_buffer(
            accumulated_out_buffer,
            self.weights.shape().to_vec(),
            device.clone(),
        );

        Ok((updated_weights, updated_accumulated))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// `AdaGrad` optimizer step - adaptive learning rate for sparse gradients
    /// **Deep Debt**: Historical optimizer, foundation for Adam/RMSprop
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as weights]
    /// - `learning_rate`: Initial step size, typically 0.01
    /// - `accumulated`: Accumulated squared gradients (None for first step)
    /// # Returns
    /// - Tuple: (`updated_weights`, `updated_accumulated`)
    /// # Example
    /// ```rust,ignore
    /// // First step
    /// let (w1, acc1) = weights.adagrad_step(&grads, 0.01, None)?;
    /// // Subsequent steps
    /// let (w2, acc2) = w1.adagrad_step(&grads, 0.01, Some(&acc1))?;
    /// ```
    /// # Note
    /// - Good for sparse gradients
    /// - Learning rate monotonically decreases
    /// - `RMSprop` and Adam address this limitation
    /// - `learning_rate` must be positive
    /// # Errors
    /// Returns [`Err`] if weights and gradients shapes differ, `learning_rate` is not positive,
    /// accumulated shape differs, buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn adagrad_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        accumulated: Option<&Self>,
    ) -> Result<(Self, Self)> {
        AdaGrad::new(self, gradients.clone(), learning_rate, accumulated.cloned())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adagrad_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.4], vec![4], device.clone())
            .await
            .unwrap();

        let (updated_weights, updated_acc) = weights.adagrad_step(&gradients, 0.01, None).unwrap();

        let result = updated_weights.to_vec().unwrap();
        let acc = updated_acc.to_vec().unwrap();

        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&x| x.is_finite()));
        assert!(acc.iter().all(|&x| x >= 0.0));
        assert!(result[0] < 1.0, "Expected descent, got {}", result[0]);
    }

    #[tokio::test]
    async fn test_adagrad_accumulation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![1.0; 4], vec![4], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.1; 4], vec![4], device.clone())
            .await
            .unwrap();

        // Step 1
        let (weights1, acc1) = weights.adagrad_step(&gradients, 0.01, None).unwrap();
        let acc_data1 = acc1.to_vec().unwrap();
        assert!(acc_data1.iter().all(|&x| x > 0.0));

        // Step 2 with accumulated state
        let (weights2, acc2) = weights1
            .adagrad_step(&gradients, 0.01, Some(&acc1))
            .unwrap();
        let result = weights2.to_vec().unwrap();
        let acc_data2 = acc2.to_vec().unwrap();

        assert!(result.iter().all(|&x| x.is_finite()));
        // Accumulated should increase
        assert!(acc_data2.iter().zip(&acc_data1).all(|(&a2, &a1)| a2 > a1));
    }

    #[tokio::test]
    async fn test_adagrad_validation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let gradients = Tensor::from_vec_on(vec![0.1; 5], vec![5], device.clone())
            .await
            .unwrap();
        let grads_correct = Tensor::from_vec_on(vec![0.1; 10], vec![10], device.clone())
            .await
            .unwrap();

        // Shape mismatch
        assert!(
            weights
                .clone()
                .adagrad_step(&gradients, 0.01, None)
                .is_err()
        );

        // Invalid learning rate
        assert!(
            weights
                .clone()
                .adagrad_step(&grads_correct, -0.01, None)
                .is_err()
        );
        assert!(weights.adagrad_step(&grads_correct, 0.0, None).is_err());
    }

    #[tokio::test]
    async fn test_adagrad_large_batch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let size = 128;
        let weights = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![0.01; size], vec![size], device.clone())
            .await
            .unwrap();

        let (updated_weights, updated_acc) = weights.adagrad_step(&gradients, 0.01, None).unwrap();

        let result = updated_weights.to_vec().unwrap();
        let acc = updated_acc.to_vec().unwrap();

        assert_eq!(result.len(), size);
        assert_eq!(acc.len(), size);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_adagrad_multi_step() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let weights = Tensor::from_vec_on(vec![10.0, 20.0], vec![2], device.clone())
            .await
            .unwrap();

        let gradients = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
            .await
            .unwrap();

        // Step 1
        let (weights1, acc1) = weights.adagrad_step(&gradients, 0.1, None).unwrap();
        let result1 = weights1.to_vec().unwrap();

        assert!(result1[0] < 10.0, "Expected descent, got {}", result1[0]);
        assert!(result1[1] < 20.0, "Expected descent, got {}", result1[1]);

        // Step 2 with accumulated state
        let (weights2, _acc2) = weights1.adagrad_step(&gradients, 0.1, Some(&acc1)).unwrap();
        let result2 = weights2.to_vec().unwrap();

        // Should continue descending
        assert!(result2[0] < result1[0]);
        assert!(result2[1] < result1[1]);
    }
}
