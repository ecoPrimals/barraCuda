// SPDX-License-Identifier: AGPL-3.0-or-later
//! Contrastive Loss - GPU-accelerated NT-Xent for self-supervised learning
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (new shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for `SimCLR`, `MoCo`)
//!
//! ## Algorithm
//!
//! ```text
//! For batch of positive pairs (z_i, z_i+batch):
//! 1. Compute cosine similarity matrix
//! 2. For each sample i:
//!    - Numerator: exp(sim(i, positive_i) / temperature)
//!    - Denominator: sum(exp(sim(i, j) / temperature)) for all j != i
//! 3. Loss = -log(numerator / denominator)
//! ```
//!
//! **Parameters**:
//! - `temperature`: Controls distribution sharpness (typically 0.1-0.5)
//!
//! **Key Properties**:
//! - Pulls positive pairs together
//! - Pushes negative pairs apart
//! - Self-supervised (no labels needed)
//! - Standard in `SimCLR`, `MoCo`
//!
//! **Used By**: `SimCLR`, `MoCo`, CLIP, self-supervised learning
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! // Embeddings: [batch*2, embed_dim] - positive pairs concatenated
//! let embeddings = Tensor::randn(vec![16, 128]).await?;  // 8 pairs, 128-dim
//!
//! let loss = embeddings.contrastive_loss(0.5)?;  // temperature=0.5
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ContrastiveLossParams {
    batch_size: u32,
    embed_dim: u32,
    temperature: f32,
    _padding: u32,
}

/// NT-Xent contrastive loss for self-supervised learning (`SimCLR`, `MoCo`).
pub struct ContrastiveLoss {
    embeddings: Tensor,
    temperature: f32,
}

impl ContrastiveLoss {
    /// Creates a new contrastive loss. Embeddings shape: [batch*2, `embed_dim`]; temperature typically 0.1–0.5.
    /// # Errors
    /// Returns [`Err`] if embeddings is not 2D, first dimension is not even (batch*2),
    /// or temperature is not positive.
    pub fn new(embeddings: Tensor, temperature: f32) -> Result<Self> {
        // Validate 2D input
        if embeddings.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "ContrastiveLoss",
                format!(
                    "embeddings must be 2D [batch*2, embed_dim], got shape {:?}",
                    embeddings.shape()
                ),
            ));
        }

        let batch_total = embeddings.shape()[0];
        if !batch_total.is_multiple_of(2) {
            return Err(BarracudaError::invalid_op(
                "ContrastiveLoss",
                format!("first dimension must be even (batch*2), got {batch_total}"),
            ));
        }

        // Validate temperature
        if temperature <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "ContrastiveLoss",
                format!("temperature must be positive, got {temperature}"),
            ));
        }

        Ok(Self {
            embeddings,
            temperature,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!(
                "../shaders/loss/contrastive_loss_f64.wgsl"
            ).to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Executes contrastive loss and returns per-sample loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.embeddings.device();
        let batch_total = self.embeddings.shape()[0];
        let embed_dim = self.embeddings.shape()[1];
        let batch_size = batch_total / 2;

        let params = ContrastiveLossParams {
            batch_size: batch_size as u32,
            embed_dim: embed_dim as u32,
            temperature: self.temperature,
            _padding: 0,
        };

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("contrastive_loss_output"),
            size: (batch_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device.create_uniform_buffer("contrastive_loss_params", &params);

        ComputeDispatch::new(device.as_ref(), "contrastive_loss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.embeddings.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(batch_size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Contrastive loss (NT-Xent) for self-supervised learning
    /// **Deep Debt**: Essential for `SimCLR`, `MoCo`, CLIP-style training
    /// # Arguments
    /// - `self`: Embeddings [batch*2, `embed_dim`] - positive pairs concatenated
    /// - `temperature`: Controls distribution sharpness (typically 0.1-0.5)
    /// # Returns
    /// - Loss tensor [`batch_size`] - per-sample losses
    /// # Example
    /// ```rust,ignore
    /// // 8 positive pairs, 128-dimensional embeddings
    /// let embeddings = Tensor::randn(vec![16, 128]).await?;
    /// // SimCLR-style: temperature=0.5
    /// let loss = embeddings.contrastive_loss(0.5)?;
    /// // MoCo-style: temperature=0.07
    /// let loss = embeddings.contrastive_loss(0.07)?;
    /// ```
    /// # Note
    /// - Input format: First `batch_size` samples paired with second `batch_size` samples
    /// - Lower temperature: Sharper distribution (more aggressive)
    /// - Higher temperature: Smoother distribution (more permissive)
    /// - Standard values: SimCLR=0.5, MoCo=0.07
    /// # Errors
    /// Returns [`Err`] if embeddings is not 2D, first dimension is not even, temperature is not positive,
    /// buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn contrastive_loss(self, temperature: f32) -> Result<Self> {
        ContrastiveLoss::new(self, temperature)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_contrastive_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 4 positive pairs (8 samples), 16-dim embeddings
        let data: Vec<f32> = (0..8 * 16).map(|i| ((i % 100) as f32) / 100.0).collect();
        let embeddings = Tensor::from_vec_on(data, vec![8, 16], device.clone())
            .await
            .unwrap();

        let loss = embeddings.contrastive_loss(0.5).unwrap();

        assert_eq!(loss.shape(), &[4]); // batch_size=4

        let data = loss.to_vec().unwrap();
        assert!(data.iter().all(|&x: &f32| x.is_finite()));
        assert!(data.iter().all(|&x: &f32| x >= 0.0)); // Loss should be non-negative
    }

    #[tokio::test]
    async fn test_contrastive_loss_similar_pairs() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Create similar positive pairs (should have relatively low loss)
        let data: Vec<f32> = (0..8 * 16)
            .map(|i| {
                let row = i / 16;
                let col = i % 16;
                // First 4 rows similar to last 4 rows
                ((row % 4) * 16 + col) as f32 / 64.0
            })
            .collect();

        let embeddings = Tensor::from_vec_on(data, vec![8, 16], device.clone())
            .await
            .unwrap();

        let loss = embeddings.contrastive_loss(0.5).unwrap();

        let data = loss.to_vec().unwrap();
        assert!(data.iter().all(|&x: &f32| x.is_finite()));
    }

    #[tokio::test]
    async fn test_contrastive_loss_temperature_effect() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let data: Vec<f32> = (0..6 * 32).map(|i| ((i % 100) as f32) / 100.0).collect();
        let embeddings = Tensor::from_vec_on(data, vec![6, 32], device.clone())
            .await
            .unwrap();

        // Lower temperature should sharpen distribution
        let loss_low_temp = embeddings.clone().contrastive_loss(0.1).unwrap();
        let loss_high_temp = embeddings.contrastive_loss(1.0).unwrap();

        let data_low = loss_low_temp.to_vec().unwrap();
        let data_high = loss_high_temp.to_vec().unwrap();

        assert!(data_low.iter().all(|&x: &f32| x.is_finite()));
        assert!(data_high.iter().all(|&x: &f32| x.is_finite()));
    }

    #[tokio::test]
    async fn test_contrastive_loss_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test odd batch size (should fail)
        let embeddings = Tensor::from_vec_on(vec![0.5; 7 * 16], vec![7, 16], device.clone())
            .await
            .unwrap();
        assert!(embeddings.contrastive_loss(0.5).is_err());

        // Test negative temperature (should fail)
        let embeddings = Tensor::from_vec_on(vec![0.5; 8 * 16], vec![8, 16], device.clone())
            .await
            .unwrap();
        assert!(embeddings.clone().contrastive_loss(-0.5).is_err());

        // Test zero temperature (should fail)
        assert!(embeddings.contrastive_loss(0.0).is_err());

        // Test 1D input (should fail)
        let embeddings = Tensor::from_vec_on(vec![0.5; 16], vec![16], device.clone())
            .await
            .unwrap();
        assert!(embeddings.contrastive_loss(0.5).is_err());
    }

    #[tokio::test]
    async fn test_contrastive_loss_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Large batch: 32 pairs (64 samples), 128-dim
        let data: Vec<f32> = (0..64 * 128).map(|i| ((i % 100) as f32) / 100.0).collect();
        let embeddings = Tensor::from_vec_on(data, vec![64, 128], device.clone())
            .await
            .unwrap();

        let loss = embeddings.contrastive_loss(0.07).unwrap(); // MoCo-style temperature

        assert_eq!(loss.shape(), &[32]);

        let data = loss.to_vec().unwrap();
        assert!(data.iter().all(|&x: &f32| x.is_finite()));
        assert!(data.iter().all(|&x: &f32| x >= 0.0));
    }
}
