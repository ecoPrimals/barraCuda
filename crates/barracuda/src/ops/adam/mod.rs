// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adam Optimizer - GPU-accelerated Adaptive Moment Estimation
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
//! m_t = β₁ * m_{t-1} + (1 - β₁) * g_t
//! v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²
//! m̂_t = m_t / (1 - β₁^t)
//! v̂_t = v_t / (1 - β₂^t)
//! θ_t = θ_{t-1} - α * m̂_t / (sqrt(v̂_t) + ε)
//! ```
//!
//! **Key Properties**:
//! - Most widely used optimizer in deep learning
//! - Combines momentum and `RMSprop`
//! - Bias correction for moving averages
//! - Works well with sparse gradients
//! - Computationally efficient
//!
//! **Parameters**:
//! - `learning_rate` (α): Step size, typically 0.001
//! - `beta1` (β₁): Exponential decay for first moment, typically 0.9
//! - `beta2` (β₂): Exponential decay for second moment, typically 0.999
//! - `epsilon` (ε): Numerical stability, typically 1e-8
//! - `step`: Current iteration number (for bias correction)
//!
//! **Used By**: Almost all modern deep learning (GPT, BERT, `ResNet`, etc.)
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
//! let (w1, m1, v1) = weights.adam_step(&gradients, 0.001, 0.9, 0.999, 1, None, None)?;
//!
//! // Subsequent steps
//! let (w2, m2, v2) = w1.adam_step(&gradients, 0.001, 0.9, 0.999, 2, Some(&m1), Some(&v1))?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Adam optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AdamParams {
    pub num_params: u32,
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step: u32,
}

/// Adam Optimizer operation
///
/// **Deep Debt**: Foundation for modern AI (GPT, BERT, `ResNet`, etc.)
pub struct Adam {
    gradients: Tensor,
    params: Tensor,
    m: Option<Tensor>,
    v: Option<Tensor>,
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    step: usize,
}

impl Adam {
    /// Create new Adam optimizer operation
    ///
    /// **Deep Debt**: Validates all inputs for shape compatibility
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes mismatch, `learning_rate` ≤ 0, betas out of
    /// [0, 1), step is 0, or m/v shapes mismatch.
    pub fn new(
        params: Tensor,
        gradients: Tensor,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        step: usize,
        m: Option<Tensor>,
        v: Option<Tensor>,
    ) -> Result<Self> {
        // Validate shapes match
        if params.shape() != gradients.shape() {
            return Err(BarracudaError::shape_mismatch(
                params.shape().to_vec(),
                gradients.shape().to_vec(),
            ));
        }

        // Validate learning rate is positive
        if learning_rate <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "adam",
                "learning_rate must be positive",
            ));
        }

        // Validate betas in valid range
        if !(0.0..1.0).contains(&beta1) {
            return Err(BarracudaError::invalid_op(
                "adam",
                "beta1 must be in range [0.0, 1.0)",
            ));
        }

        if !(0.0..1.0).contains(&beta2) {
            return Err(BarracudaError::invalid_op(
                "adam",
                "beta2 must be in range [0.0, 1.0)",
            ));
        }

        // Validate step is positive
        if step == 0 {
            return Err(BarracudaError::invalid_op(
                "adam",
                "step must be >= 1 (starts at 1, not 0)",
            ));
        }

        // Validate m and v shapes if provided
        if let Some(ref m_tensor) = m {
            if m_tensor.shape() != params.shape() {
                return Err(BarracudaError::shape_mismatch(
                    m_tensor.shape().to_vec(),
                    params.shape().to_vec(),
                ));
            }
        }

        if let Some(ref v_tensor) = v {
            if v_tensor.shape() != params.shape() {
                return Err(BarracudaError::shape_mismatch(
                    v_tensor.shape().to_vec(),
                    params.shape().to_vec(),
                ));
            }
        }

        Ok(Self {
            gradients,
            params,
            m,
            v,
            learning_rate,
            beta1,
            beta2,
            step,
        })
    }

    /// WGSL shader source
    pub(super) fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../../shaders/optimizer/adam_f64.wgsl").to_string()
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
    pub(super) fn m(&self) -> Option<&Tensor> {
        self.m.as_ref()
    }

    /// Get v tensor
    pub(super) fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
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

    /// Get step
    pub(super) fn step(&self) -> usize {
        self.step
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION (MODERN IDIOMATIC RUST)
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Adam optimizer step - most widely used optimizer in deep learning
    ///
    /// **Deep Debt**: Foundation for modern AI (GPT, BERT, `ResNet`, etc.)
    ///
    /// # Arguments
    /// - `gradients`: Gradient tensor [same shape as params]
    /// - `learning_rate`: Step size, typically 0.001
    /// - `beta1`: Exponential decay for first moment, typically 0.9
    /// - `beta2`: Exponential decay for second moment, typically 0.999
    /// - `step`: Current iteration (starts at 1, not 0)
    /// - `m`: First moment estimate (None for first step)
    /// - `v`: Second moment estimate (None for first step)
    ///
    /// # Returns
    /// - Tuple: (`updated_params`, `updated_m`, `updated_v`)
    ///
    /// # Example
    /// ```rust,ignore
    /// // First step
    /// let (w1, m1, v1) = weights.adam_step(&grads, 0.001, 0.9, 0.999, 1, None, None)?;
    ///
    /// // Subsequent steps
    /// let (w2, m2, v2) = w1.adam_step(&grads, 0.001, 0.9, 0.999, 2, Some(&m1), Some(&v1))?;
    /// ```
    ///
    /// # Note
    /// - Most widely used optimizer in deep learning
    /// - `learning_rate` must be positive
    /// - beta1, beta2 must be in [0.0, 1.0)
    /// - step must start at 1 (not 0)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn adam_step(
        self,
        gradients: &Self,
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        step: usize,
        m: Option<&Self>,
        v: Option<&Self>,
    ) -> Result<(Self, Self, Self)> {
        Adam::new(
            self,
            gradients.clone(),
            learning_rate,
            beta1,
            beta2,
            step,
            m.cloned(),
            v.cloned(),
        )?
        .execute()
    }
}
