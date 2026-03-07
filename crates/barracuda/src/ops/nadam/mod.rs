// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NAdam` Optimizer - GPU-accelerated Nesterov-accelerated Adam
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
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//! use barracuda::ops::adamw::AdamConfig;
//!
//! let config = AdamConfig::new(0.001);
//! let (new_weights, new_m, new_v) = weights.nadam(&gradients, &m, &v, &config, 1)?;
//! ```

mod compute;

#[cfg(test)]
mod tests;

use crate::error::{BarracudaError, Result};
use crate::ops::adamw::AdamConfig;
use crate::tensor::Tensor;

/// `NAdam` optimizer parameters for WGSL shader
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NadamParams {
    pub learning_rate: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub weight_decay: f32,
    pub step: u32,
    pub _padding: [u32; 2],
}

/// `NAdam` Optimizer operation
pub struct Nadam {
    weights: Tensor,
    gradients: Tensor,
    m: Tensor,
    v: Tensor,
    config: AdamConfig,
    step: u32,
}

impl Nadam {
    /// Create new `NAdam` optimizer operation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if shapes mismatch, hyperparameters are invalid,
    /// or step is 0.
    pub fn new(
        weights: Tensor,
        gradients: Tensor,
        m: Tensor,
        v: Tensor,
        config: &AdamConfig,
        step: u32,
    ) -> Result<Self> {
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
        config.validate("NAdam")?;
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
            config: *config,
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

    pub(super) fn weights(&self) -> &Tensor {
        &self.weights
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
    /// `NAdam` optimizer step (Nesterov-accelerated Adam).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn nadam(
        self,
        gradients: &Self,
        m: &Self,
        v: &Self,
        config: &AdamConfig,
        step: u32,
    ) -> Result<(Self, Self, Self)> {
        Nadam::new(self, gradients.clone(), m.clone(), v.clone(), config, step)?.execute()
    }
}
