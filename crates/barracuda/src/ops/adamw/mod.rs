// SPDX-License-Identifier: AGPL-3.0-or-later
//! `AdamW` Optimizer - GPU-accelerated Adam with Decoupled Weight Decay
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
//! **Key Difference from Adam**: Weight decay is decoupled from gradient update.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//! use barracuda::ops::adamw::AdamConfig;
//!
//! let config = AdamConfig::new(0.001).weight_decay(0.01);
//! let (new_weights, new_m, new_v) = weights.adamw(&gradients, &m, &v, &config, 1)?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Shared hyperparameters for Adam-family optimizers (`AdamW`, `NAdam`, etc.).
///
/// Use the builder pattern to construct, with sensible defaults for `beta1`,
/// `beta2`, `epsilon`, and `weight_decay`.
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Learning rate.
    pub learning_rate: f32,
    /// First moment decay (default 0.9).
    pub beta1: f32,
    /// Second moment decay (default 0.999).
    pub beta2: f32,
    /// Numerical stability term (default 1e-8).
    pub epsilon: f32,
    /// Decoupled weight decay (default 0.0 = none).
    pub weight_decay: f32,
}

impl AdamConfig {
    /// Create a new config with the given learning rate and standard defaults.
    #[must_use]
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
        }
    }

    /// Set momentum decay rates (default: beta1=0.9, beta2=0.999).
    #[must_use]
    pub fn beta(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Set numerical stability term (default: 1e-8).
    #[must_use]
    pub fn epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Set decoupled weight decay (default: 0.0 = none).
    #[must_use]
    pub fn weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Validate hyperparameters.
    pub(crate) fn validate(&self, op_name: &str) -> Result<()> {
        if !(0.0..1.0).contains(&self.beta1) {
            return Err(BarracudaError::invalid_op(
                op_name,
                format!("beta1 must be in [0, 1), got {}", self.beta1),
            ));
        }
        if !(0.0..1.0).contains(&self.beta2) {
            return Err(BarracudaError::invalid_op(
                op_name,
                format!("beta2 must be in [0, 1), got {}", self.beta2),
            ));
        }
        if self.epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                op_name,
                format!("epsilon must be positive, got {}", self.epsilon),
            ));
        }
        Ok(())
    }
}

/// `AdamW` optimizer parameters for WGSL shader
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

/// `AdamW` Optimizer operation
pub struct AdamW {
    params: Tensor,
    gradients: Tensor,
    m: Tensor,
    v: Tensor,
    config: AdamConfig,
    step: u32,
}

impl AdamW {
    /// Create new `AdamW` optimizer operation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes mismatch, hyperparameters are invalid,
    /// or step is 0.
    pub fn new(
        params: Tensor,
        gradients: Tensor,
        m: Tensor,
        v: Tensor,
        config: &AdamConfig,
        step: u32,
    ) -> Result<Self> {
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
        config.validate("AdamW")?;
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
            config: *config,
            step,
        })
    }

    /// WGSL shader source
    pub(super) fn shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../../shaders/optimizer/adamw_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    pub(super) fn params(&self) -> &Tensor {
        &self.params
    }
    pub(super) fn gradients(&self) -> &Tensor {
        &self.gradients
    }
    pub(super) fn m(&self) -> &Tensor {
        &self.m
    }
    pub(super) fn v(&self) -> &Tensor {
        &self.v
    }
    pub(super) fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }
    pub(super) fn beta1(&self) -> f32 {
        self.config.beta1
    }
    pub(super) fn beta2(&self) -> f32 {
        self.config.beta2
    }
    pub(super) fn epsilon(&self) -> f32 {
        self.config.epsilon
    }
    pub(super) fn weight_decay(&self) -> f32 {
        self.config.weight_decay
    }
    pub(super) fn step(&self) -> u32 {
        self.step
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// `AdamW` optimizer step (Adam with Decoupled Weight Decay).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn adamw(
        self,
        gradients: &Self,
        m: &Self,
        v: &Self,
        config: &AdamConfig,
        step: u32,
    ) -> Result<(Self, Self, Self)> {
        AdamW::new(self, gradients.clone(), m.clone(), v.clone(), config, step)?.execute()
    }
}
