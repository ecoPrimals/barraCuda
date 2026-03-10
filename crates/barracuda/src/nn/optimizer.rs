// SPDX-License-Identifier: AGPL-3.0-only
//! Optimizer Implementations
//!
//! Various optimization algorithms for neural network training.
//! Deep Debt compliant: Runtime parameter configuration.

/// Optimizer types (capability-based)
#[derive(Debug, Clone)]
pub enum Optimizer {
    /// Adam optimizer with learning rate, betas, and epsilon.
    Adam {
        /// Learning rate.
        lr: f32,
        /// Beta parameters (typically 0.9, 0.999).
        betas: (f32, f32),
        /// Numerical stability epsilon.
        eps: f32,
    },
    /// `AdaGrad` optimizer.
    AdaGrad {
        /// Learning rate.
        lr: f32,
        /// Numerical stability epsilon.
        eps: f32,
    },
    /// `AdaDelta` optimizer.
    AdaDelta {
        /// Decay factor rho.
        rho: f32,
        /// Numerical stability epsilon.
        eps: f32,
    },
    /// SGD with momentum.
    SGD {
        /// Learning rate.
        lr: f32,
        /// Momentum factor.
        momentum: f32,
    },
}
