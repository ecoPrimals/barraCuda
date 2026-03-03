// SPDX-License-Identifier: AGPL-3.0-or-later
//! RMSprop Optimizer - GPU-accelerated Root Mean Square Propagation
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
//! E[g²] = α * E[g²] + (1 - α) * g²
//! w = w - lr * g / (sqrt(E[g²]) + ε)
//! ```
//!
//! **Key Properties**:
//! - Adaptive learning rate per parameter
//! - Uses moving average of squared gradients
//! - More stable than AdaGrad (doesn't monotonically decrease)
//! - Popular for RNNs and non-stationary problems
//!
//! **Parameters**:
//! - `learning_rate`: Step size, typically 0.001-0.01
//! - `alpha` (α): Decay rate for moving average, typically 0.99
//! - `epsilon` (ε): Numerical stability constant, typically 1e-8
//!
//! **Used By**: RNNs, non-stationary optimization problems
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
//! let (w1, sq_avg1) = weights.rmsprop_step(&gradients, 0.001, 0.99, None)?;
//!
//! // Subsequent steps
//! let (w2, sq_avg2) = w1.rmsprop_step(&gradients, 0.001, 0.99, Some(&sq_avg1))?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// RMSprop optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RMSpropParams {
    pub learning_rate: f32,
    pub alpha: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
}

/// RMSprop Optimizer operation
///
/// **Deep Debt**: Uses existing WGSL shader for adaptive learning rate optimization
pub struct RMSprop {
    weights: Tensor,
    gradients: Tensor,
    sq_avg: Option<Tensor>,
    learning_rate: f32,
    alpha: f32,
}

impl RMSprop {
    /// Create new RMSprop optimizer operation
    ///
    /// **Deep Debt**: Validates all inputs for shape compatibility
    pub fn new(
        weights: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        alpha: f32,
        sq_avg: Option<Tensor>,
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
                "rmsprop",
                "learning_rate must be positive",
            ));
        }

        // Validate alpha in valid range
        if !(0.0..=1.0).contains(&alpha) {
            return Err(BarracudaError::invalid_op(
                "rmsprop",
                "alpha must be in range [0.0, 1.0]",
            ));
        }

        // Validate sq_avg shape if provided
        if let Some(ref sq) = sq_avg {
            if sq.shape() != weights.shape() {
                return Err(BarracudaError::shape_mismatch(
                    sq.shape().to_vec(),
                    weights.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            weights,
            gradients,
            sq_avg,
            learning_rate,
            alpha,
        })
    }

    /// WGSL shader source
    pub(super) fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../../shaders/optimizer/rmsprop_f64.wgsl"
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

    /// Get sq_avg tensor
    pub(super) fn sq_avg(&self) -> Option<&Tensor> {
        self.sq_avg.as_ref()
    }

    /// Get learning rate
    pub(super) fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get alpha
    pub(super) fn alpha(&self) -> f32 {
        self.alpha
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// RMSprop optimizer step - adaptive learning rate optimizer
    ///
    /// **Deep Debt**: Essential for RNNs and non-stationary problems
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as weights]
    /// - `learning_rate`: Step size, typically 0.001-0.01
    /// - `alpha`: Decay rate for moving average, typically 0.99
    /// - `sq_avg`: Accumulated squared gradients (None for first step)
    ///
    /// # Returns
    /// - Tuple: (updated_weights, updated_sq_avg)
    ///
    /// # Example
    /// ```rust,ignore
    /// // First step
    /// let (w1, sq1) = weights.rmsprop_step(&grads, 0.001, 0.99, None)?;
    ///
    /// // Subsequent steps
    /// let (w2, sq2) = w1.rmsprop_step(&grads, 0.001, 0.99, Some(&sq1))?;
    /// ```
    ///
    /// # Note
    /// - Adaptive learning rate per parameter
    /// - Popular for RNNs
    /// - learning_rate must be positive
    /// - alpha must be in [0.0, 1.0]
    pub fn rmsprop_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        alpha: f32,
        sq_avg: Option<&Self>,
    ) -> Result<(Self, Self)> {
        RMSprop::new(
            self,
            gradients.clone(),
            learning_rate,
            alpha,
            sq_avg.cloned(),
        )?
        .execute()
    }
}
