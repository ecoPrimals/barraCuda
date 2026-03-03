//! SGD Optimizer - GPU-accelerated Stochastic Gradient Descent
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
//! Without momentum:
//! w = w - lr * (g + weight_decay * w)
//!
//! With momentum:
//! v = momentum * v + g
//! w = w - lr * v
//! ```
//!
//! **Key Properties**:
//! - Foundation optimizer for deep learning
//! - Optional momentum for faster convergence
//! - Optional weight decay for regularization
//! - Simple and robust
//!
//! **Parameters**:
//! - `learning_rate`: Step size, typically 0.01-0.1
//! - `momentum`: Momentum factor, typically 0.9 (0.0 = no momentum)
//! - `weight_decay`: L2 regularization, typically 0.0001-0.001
//!
//! **Used By**: All deep learning training (foundational optimizer)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let weights = Tensor::randn(vec![1000]).await?;
//! let gradients = Tensor::randn(vec![1000]).await?;
//!
//! // Without momentum
//! let (updated_weights, _) =
//!     weights.sgd_step(&gradients, 0.01, 0.0, 0.0, None)?;
//!
//! // With momentum
//! let (w1, v1) = weights.sgd_step(&gradients, 0.01, 0.9, 0.0, None)?;
//! let (w2, v2) = w1.sgd_step(&gradients, 0.01, 0.9, 0.0, v1.as_ref())?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

/// f64 is the canonical source.
const SHADER_F64: &str = include_str!("../../shaders/optimizer/sgd_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// SGD optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SGDParams {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub dampening: f32,
}

/// SGD Optimizer operation
///
/// **Deep Debt**: Foundation optimizer for all deep learning training
pub struct SGD {
    weights: Tensor,
    gradients: Tensor,
    velocity: Option<Tensor>,
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
}

impl SGD {
    /// Create new SGD optimizer operation
    ///
    /// **Deep Debt**: Validates all inputs for shape compatibility
    pub fn new(
        weights: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        velocity: Option<Tensor>,
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
                "sgd",
                "learning_rate must be positive",
            ));
        }

        // Validate momentum in valid range
        if !(0.0..=1.0).contains(&momentum) {
            return Err(BarracudaError::invalid_op(
                "sgd",
                "momentum must be in range [0.0, 1.0]",
            ));
        }

        // Validate weight_decay is non-negative
        if weight_decay < 0.0 {
            return Err(BarracudaError::invalid_op(
                "sgd",
                "weight_decay must be non-negative",
            ));
        }

        // Validate velocity shape if provided
        if let Some(ref v) = velocity {
            if v.shape() != weights.shape() {
                return Err(BarracudaError::shape_mismatch(
                    v.shape().to_vec(),
                    weights.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            weights,
            gradients,
            velocity,
            learning_rate,
            momentum,
            weight_decay,
        })
    }

    /// WGSL shader source (f64 canonical, f32 via downcast)
    pub(super) fn shader() -> &'static str {
        &SHADER_F32
    }

    /// Get weights tensor
    pub(super) fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get gradients tensor
    pub(super) fn gradients(&self) -> &Tensor {
        &self.gradients
    }

    /// Get velocity tensor
    pub(super) fn velocity(&self) -> Option<&Tensor> {
        self.velocity.as_ref()
    }

    /// Get learning rate
    pub(super) fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Get momentum
    pub(super) fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Get weight decay
    pub(super) fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// SGD optimizer step - foundational gradient descent optimizer
    ///
    /// **Deep Debt**: Foundation for all deep learning training
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as weights]
    /// - `learning_rate`: Step size, typically 0.01-0.1
    /// - `momentum`: Momentum factor 0.0-1.0, typically 0.9 (0.0 = no momentum)
    /// - `weight_decay`: L2 regularization, typically 0.0001-0.001
    /// - `velocity`: Momentum velocity (None for first step)
    ///
    /// # Returns
    /// - Tuple: (updated_weights, updated_velocity)
    ///
    /// # Example
    /// ```rust,ignore
    /// // Without momentum
    /// let (w1, _) = weights.sgd_step(&grads, 0.01, 0.0, 0.0, None)?;
    ///
    /// // With momentum
    /// let (w1, v1) = weights.sgd_step(&grads, 0.01, 0.9, 0.0, None)?;
    /// let (w2, v2) = w1.sgd_step(&grads, 0.01, 0.9, 0.0, v1.as_ref())?;
    /// ```
    ///
    /// # Note
    /// - Foundation optimizer for deep learning
    /// - learning_rate must be positive
    /// - momentum must be in [0.0, 1.0]
    /// - weight_decay must be non-negative
    pub fn sgd_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        velocity: Option<&Self>,
    ) -> Result<(Self, Option<Self>)> {
        SGD::new(
            self,
            gradients.clone(),
            learning_rate,
            momentum,
            weight_decay,
            velocity.cloned(),
        )?
        .execute()
    }
}
