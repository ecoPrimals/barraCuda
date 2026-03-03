// SPDX-License-Identifier: AGPL-3.0-or-later
//! AdaDelta Optimizer - GPU-accelerated adaptive learning rate optimizer
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
//! E[g²] = ρ * E[g²] + (1 - ρ) * g²
//! RMS[g] = sqrt(E[g²] + ε)
//! RMS[Δ] = sqrt(E[Δ²] + ε)
//! Δw = -(RMS[Δ] / RMS[g]) * g
//! w = w + Δw
//! E[Δ²] = ρ * E[Δ²] + (1 - ρ) * Δw²
//! ```
//!
//! **Key Properties**:
//! - No learning rate hyperparameter needed!
//! - Adapts learning rate per parameter
//! - More stable than AdaGrad (doesn't monotonically decrease)
//! - Uses moving average of gradients and updates
//!
//! **Parameters**:
//! - `rho` (ρ): Decay rate for moving averages, typically 0.95
//! - `epsilon` (ε): Numerical stability constant, typically 1e-6
//!
//! **Used By**: When you want to avoid tuning learning rates
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let weights = Tensor::randn(vec![1000]).await?;
//! let gradients = Tensor::randn(vec![1000]).await?;
//!
//! // First step (no accumulated state)
//! let (updated_weights, acc_grad, acc_delta) =
//!     weights.adadelta_step(&gradients, 0.95, None, None)?;
//!
//! // Subsequent steps (with accumulated state)
//! let (updated_weights2, acc_grad2, acc_delta2) =
//!     updated_weights.adadelta_step(
//!         &gradients,
//!         0.95,
//!         Some(&acc_grad),
//!         Some(&acc_delta),
//!     )?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// AdaDelta optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AdaDeltaParams {
    pub rho: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub _padding: u32,
}

/// AdaDelta Optimizer operation
///
/// **Deep Debt**: Uses WGSL shader with adaptive learning rate
pub struct AdaDelta {
    weights: Tensor,
    gradients: Tensor,
    acc_grad: Option<Tensor>,
    acc_delta: Option<Tensor>,
    rho: f32,
}

impl AdaDelta {
    /// Create new AdaDelta optimizer operation
    ///
    /// **Deep Debt**: Validates all inputs for shape compatibility
    pub fn new(
        weights: Tensor,
        gradients: Tensor,
        rho: f32,
        acc_grad: Option<Tensor>,
        acc_delta: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if weights.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                weights.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate rho in valid range
        if !(0.0..=1.0).contains(&rho) {
            return Err(BarracudaError::invalid_op(
                "adadelta",
                "rho must be in range [0.0, 1.0]",
            ));
        }

        // Validate accumulator shapes if provided
        if let Some(ref ag) = acc_grad {
            if ag.shape() != weights.shape() {
                return Err(BarracudaError::shape_mismatch(
                    ag.shape().to_vec(),
                    weights.shape().to_vec(),
                ));
            }
        }

        if let Some(ref ad) = acc_delta {
            if ad.shape() != weights.shape() {
                return Err(BarracudaError::shape_mismatch(
                    ad.shape().to_vec(),
                    weights.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            weights,
            gradients,
            acc_grad,
            acc_delta,
            rho,
        })
    }

    /// WGSL shader source
    pub(super) fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../../shaders/optimizer/adadelta_f64.wgsl"
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

    /// Get accumulated gradients tensor
    pub(super) fn acc_grad(&self) -> Option<&Tensor> {
        self.acc_grad.as_ref()
    }

    /// Get accumulated delta tensor
    pub(super) fn acc_delta(&self) -> Option<&Tensor> {
        self.acc_delta.as_ref()
    }

    /// Get rho parameter
    pub(super) fn rho(&self) -> f32 {
        self.rho
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// AdaDelta optimizer step - adaptive learning rate without lr hyperparameter
    ///
    /// **Deep Debt**: Essential for training without tuning learning rates
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as weights]
    /// - `rho`: Decay rate for moving averages, typically 0.95
    /// - `acc_grad`: Accumulated squared gradients (None for first step)
    /// - `acc_delta`: Accumulated squared deltas (None for first step)
    ///
    /// # Returns
    /// - Tuple: (updated_weights, updated_acc_grad, updated_acc_delta)
    ///
    /// # Example
    /// ```rust,ignore
    /// // First step
    /// let (w1, ag1, ad1) = weights.adadelta_step(&grads, 0.95, None, None)?;
    ///
    /// // Subsequent steps
    /// let (w2, ag2, ad2) = w1.adadelta_step(&grads, 0.95, Some(&ag1), Some(&ad1))?;
    /// ```
    ///
    /// # Note
    /// - No learning rate hyperparameter needed!
    /// - More stable than AdaGrad
    /// - rho should be in [0.0, 1.0], typically 0.95
    pub fn adadelta_step(
        self,
        gradients: &Self,
        rho: f32,
        acc_grad: Option<&Self>,
        acc_delta: Option<&Self>,
    ) -> Result<(Self, Self, Self)> {
        AdaDelta::new(
            self,
            gradients.clone(),
            rho,
            acc_grad.cloned(),
            acc_delta.cloned(),
        )?
        .execute()
    }
}
