// SPDX-License-Identifier: AGPL-3.0-or-later
//! AdamW Optimizer - GPU-accelerated Adam with Decoupled Weight Decay
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (new shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready optimizer)
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)
//!
//! ## Algorithm
//!
//! ```text
//! m = beta1 * m + (1 - beta1) * gradient
//! v = beta2 * v + (1 - beta2) * gradient²
//! m_hat = m / (1 - beta1^t)
//! v_hat = v / (1 - beta2^t)
//! param = param - lr * m_hat / (sqrt(v_hat) + epsilon) - lr * wd * param
//! ```
//!
//! **Key Difference from Adam**: Weight decay is decoupled from gradient update!
//!
//! **Implementation**: Single-pass GPU optimizer with decoupled weight decay
//!
//! **Key Properties**:
//! - Decoupled weight decay (superior to L2 regularization)
//! - Works better with adaptive learning rates
//! - Standard in modern transformers (BERT, GPT, etc.)
//! - Automatic bias correction
//!
//! **Used By**: Modern deep learning, large language models, SOTA training
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let weights = Tensor::randn(vec![1000, 512]).await?;
//! let gradients = Tensor::randn(vec![1000, 512]).await?;
//! let m = Tensor::zeros(vec![1000, 512]).await?;
//! let v = Tensor::zeros(vec![1000, 512]).await?;
//!
//! let (new_weights, new_m, new_v) = weights.adamw(
//!     &gradients,
//!     &m,
//!     &v,
//!     0.001,  // learning_rate
//!     0.9,    // beta1
//!     0.999,  // beta2
//!     1e-8,   // epsilon
//!     0.01,   // weight_decay (decoupled!)
//!     1,      // step
//! )?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// AdamW optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AdamWParams {
    pub num_params: u32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step: u32,
}

/// AdamW Optimizer operation
///
/// **Deep Debt**: Uses new WGSL shader with decoupled weight decay
pub struct AdamW {
    params: Tensor,
    gradients: Tensor,
    m: Tensor,
    v: Tensor,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: f32,
    step: u32,
}

impl AdamW {
    /// Create new AdamW optimizer operation
    ///
    /// **Deep Debt**: Validates all inputs for shape compatibility
    #[expect(clippy::too_many_arguments, reason = "API")]
    pub fn new(
        params: Tensor,
        gradients: Tensor,
        m: Tensor,
        v: Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<Self> {
        // Validate shapes match
        if params.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                params.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }
        if params.shape() != m.shape() {
            return Err(BarracudaError::shape_mismatch(
                params.shape().to_vec(),
                m.shape().to_vec(),
            ));
        }
        if params.shape() != v.shape() {
            return Err(BarracudaError::shape_mismatch(
                params.shape().to_vec(),
                v.shape().to_vec(),
            ));
        }

        // Validate hyperparameters
        if !(0.0..1.0).contains(&beta1) {
            return Err(BarracudaError::invalid_op(
                "AdamW",
                format!("beta1 must be in [0, 1), got {beta1}"),
            ));
        }
        if !(0.0..1.0).contains(&beta2) {
            return Err(BarracudaError::invalid_op(
                "AdamW",
                format!("beta2 must be in [0, 1), got {beta2}"),
            ));
        }
        if epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "AdamW",
                format!("epsilon must be positive, got {epsilon}"),
            ));
        }
        if step == 0 {
            return Err(BarracudaError::invalid_op(
                "AdamW",
                "step must be >= 1 for bias correction",
            ));
        }

        Ok(Self {
            params,
            gradients,
            m,
            v,
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
        })
    }

    /// WGSL shader source
    pub(super) fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../../shaders/optimizer/adamw_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Get params tensor
    pub(super) fn params(&self) -> &Tensor {
        &self.params
    }

    /// Get gradients tensor
    pub(super) fn gradients(&self) -> &Tensor {
        &self.gradients
    }

    /// Get m tensor
    pub(super) fn m(&self) -> &Tensor {
        &self.m
    }

    /// Get v tensor
    pub(super) fn v(&self) -> &Tensor {
        &self.v
    }

    /// Get learning rate
    pub(super) fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get beta1
    pub(super) fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Get beta2
    pub(super) fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Get epsilon
    pub(super) fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get weight decay
    pub(super) fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Get step
    pub(super) fn step(&self) -> u32 {
        self.step
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// AdamW optimizer step (Adam with Decoupled Weight Decay)
    ///
    /// **Deep Debt**: Production-ready optimizer with decoupled weight decay
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as params]
    /// - `m`: First moment estimate [same shape as params]
    /// - `v`: Second moment estimate [same shape as params]
    /// - `learning_rate`: Learning rate (e.g., 0.001)
    /// - `beta1`: First moment decay (typically 0.9)
    /// - `beta2`: Second moment decay (typically 0.999)
    /// - `epsilon`: Numerical stability (typically 1e-8)
    /// - `weight_decay`: Decoupled weight decay (typically 0.01, 0.0 = none)
    /// - `step`: Current step number (for bias correction, must be >= 1)
    ///
    /// # Returns
    /// - `(new_params, new_m, new_v)`: Updated parameters and moments
    ///
    /// # Example
    /// ```rust,ignore
    /// let (p, m, v) = params.adamw(&grad, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.01, 1)?;
    /// ```
    ///
    /// # Note
    /// AdamW is superior to Adam for large models due to decoupled weight decay!
    #[expect(clippy::too_many_arguments, reason = "API")]
    pub fn adamw(
        self,
        gradients: &Self,
        m: &Self,
        v: &Self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        step: u32,
    ) -> Result<(Self, Self, Self)> {
        AdamW::new(
            self,
            gradients.clone(),
            m.clone(),
            v.clone(),
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            step,
        )?
        .execute()
    }
}
