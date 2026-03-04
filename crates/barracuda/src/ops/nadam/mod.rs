// SPDX-License-Identifier: AGPL-3.0-or-later
//! NAdam Optimizer - GPU-accelerated Nesterov-accelerated Adam
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (uses existing shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready optimizer)
//!
//! ## Algorithm
//!
//! ```text
//! m = beta1 * m + (1 - beta1) * gradient
//! v = beta2 * v + (1 - beta2) * gradient²
//! m_hat = m / (1 - beta1^t)
//! v_hat = v / (1 - beta2^t)
//! gradient_nesterov = (beta1 * m_hat + (1 - beta1) * gradient) / (1 - beta1^t)
//! weight = weight - learning_rate * gradient_nesterov / (sqrt(v_hat) + epsilon)
//! ```
//!
//! **Implementation**: Single-pass GPU optimizer with Nesterov momentum
//!
//! **Key Properties**:
//! - Combines Adam with Nesterov momentum
//! - Faster convergence than standard Adam
//! - Automatic bias correction
//! - Optional weight decay (L2 regularization)
//!
//! **Used By**: Modern deep learning training, faster than Adam
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
//! let (new_weights, new_m, new_v) = weights.nadam(
//!     &gradients,
//!     &m,
//!     &v,
//!     0.001,  // learning_rate
//!     0.9,    // beta1
//!     0.999,  // beta2
//!     1e-8,   // epsilon
//!     0.0,    // weight_decay
//!     1,      // step
//! )?;
//! ```

mod compute;

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// NAdam optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NadamParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step: u32,
    pub _padding: [u32; 2], // Explicit padding for 16-byte alignment
}

/// NAdam Optimizer operation
///
/// **Deep Debt**: Uses existing WGSL shader with Nesterov momentum
pub struct Nadam {
    weights: Tensor,
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

impl Nadam {
    /// Create new NAdam optimizer operation
    ///
    /// **Deep Debt**: Validates all inputs for shape compatibility
    #[expect(clippy::too_many_arguments, reason = "API")]
    pub fn new(
        weights: Tensor,
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
        if weights.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                weights.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }
        if weights.shape() != m.shape() {
            return Err(BarracudaError::shape_mismatch(
                weights.shape().to_vec(),
                m.shape().to_vec(),
            ));
        }
        if weights.shape() != v.shape() {
            return Err(BarracudaError::shape_mismatch(
                weights.shape().to_vec(),
                v.shape().to_vec(),
            ));
        }

        // Validate hyperparameters
        if !(0.0..1.0).contains(&beta1) {
            return Err(BarracudaError::invalid_op(
                "NAdam",
                format!("beta1 must be in [0, 1), got {beta1}"),
            ));
        }
        if !(0.0..1.0).contains(&beta2) {
            return Err(BarracudaError::invalid_op(
                "NAdam",
                format!("beta2 must be in [0, 1), got {beta2}"),
            ));
        }
        if epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "NAdam",
                format!("epsilon must be positive, got {epsilon}"),
            ));
        }
        if step == 0 {
            return Err(BarracudaError::invalid_op(
                "NAdam",
                "step must be >= 1 for bias correction",
            ));
        }

        Ok(Self {
            weights,
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
                    "../../shaders/optimizer/nadam_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Get weights tensor
    pub(super) fn weights(&self) -> &Tensor {
        &self.weights
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
    /// NAdam optimizer step (Nesterov-accelerated Adam)
    ///
    /// **Deep Debt**: Production-ready optimizer with Nesterov momentum
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as weights]
    /// - `m`: First moment estimate [same shape as weights]
    /// - `v`: Second moment estimate [same shape as weights]
    /// - `learning_rate`: Learning rate (e.g., 0.001)
    /// - `beta1`: First moment decay (typically 0.9)
    /// - `beta2`: Second moment decay (typically 0.999)
    /// - `epsilon`: Numerical stability (typically 1e-8)
    /// - `weight_decay`: L2 regularization (0.0 = none)
    /// - `step`: Current step number (for bias correction, must be >= 1)
    ///
    /// # Returns
    /// - `(new_weights, new_m, new_v)`: Updated parameters and moments
    ///
    /// # Example
    /// ```rust,ignore
    /// let (w, m, v) = weights.nadam(&grad, &m, &v, 0.001, 0.9, 0.999, 1e-8, 0.0, 1)?;
    /// ```
    #[expect(clippy::too_many_arguments, reason = "API")]
    pub fn nadam(
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
        Nadam::new(
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
